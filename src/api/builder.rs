//! Builds a frozen bitmap from values pushed in strictly ascending order.

use crate::api::bitmap::{result_buf, FrozenBitmap};
use crate::container::{Bitmap, Data, Run};
use crate::format::*;

/// Accumulates ascending `u32`s, then serializes to a [`FrozenBitmap`].
///
/// ```
/// # use frostbit::FrozenBitmapBuilder;
/// let mut b = FrozenBitmapBuilder::new();
/// b.extend_sorted([10, 20, 70_000]);
/// let bm = b.finish();
/// ```
pub struct FrozenBitmapBuilder {
    containers: Vec<Built>,
    cur_key: u16,
    /// Lows of the open container. Reused across containers, so it is
    /// allocated once (at most 64K `u16`) for the builder's lifetime.
    cur: Vec<u16>,
    have_cur: bool,
    total: u64,
}

struct Built {
    key: u16,
    typ: u8,
    card: u32,
    data: ContainerData,
}

/// A finished container in its winning representation: the owned counterpart of
/// [`Data`], held until `finish` writes it into the output.
enum ContainerData {
    Array(Vec<u16>),
    Bitmap(Box<Bitmap>),
    Run(Vec<Run>),
}

impl ContainerData {
    /// Borrow as the typed view the kernels read.
    fn view(&self) -> Data<'_> {
        match self {
            ContainerData::Array(vals) => Data::Array(vals),
            ContainerData::Bitmap(words) => Data::Bitmap(words),
            ContainerData::Run(runs) => Data::Run(runs),
        }
    }

    /// Serialized payload size.
    fn byte_len(&self) -> usize {
        match self {
            ContainerData::Array(vals) => vals.len() * 2,
            ContainerData::Bitmap(_) => BITMAP_BYTES,
            ContainerData::Run(runs) => run_bytes(runs.len()),
        }
    }

    /// Write the payload into `dst`, exactly `byte_len` bytes.
    fn write(&self, dst: &mut [u8]) {
        match self {
            ContainerData::Array(vals) => {
                for (j, &v) in vals.iter().enumerate() {
                    write_u16(dst, j * 2, v);
                }
            }
            ContainerData::Bitmap(words) => {
                for (j, &w) in words.iter().enumerate() {
                    write_u64(dst, j * 8, w);
                }
            }
            ContainerData::Run(runs) => {
                write_u16(dst, 0, runs.len() as u16);
                for (j, r) in runs.iter().enumerate() {
                    write_u16(dst, 2 + j * 4, r.start);
                    write_u16(dst, 2 + j * 4 + 2, r.len);
                }
            }
        }
    }
}

impl FrozenBitmapBuilder {
    /// A new, empty builder. Reserves the array/bitmap break-even up front.
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
            cur_key: 0,
            cur: Vec::with_capacity(ARRAY_MAX_SIZE),
            have_cur: false,
            total: 0,
        }
    }

    /// Push a value strictly greater than every prior value.
    ///
    /// # Panics
    /// If `value` is not strictly greater than the previous push.
    pub fn push(&mut self, value: u32) {
        let key = (value >> 16) as u16;
        let lo = (value & 0xFFFF) as u16;
        if self.have_cur && key == self.cur_key {
            assert!(
                lo > *self.cur.last().unwrap(),
                "values must be strictly ascending"
            );
            self.cur.push(lo);
        } else {
            if self.have_cur {
                assert!(key > self.cur_key, "values must be strictly ascending");
                self.flush();
            }
            self.cur_key = key;
            self.cur.push(lo);
            self.have_cur = true;
        }
    }

    /// Push every value from `iter`. Each must be strictly greater than all
    /// prior values across the whole build; a non-ascending value panics (see
    /// [`push`](Self::push)). There is no `Extend` impl for this reason.
    pub fn extend_sorted(&mut self, iter: impl IntoIterator<Item = u32>) {
        for v in iter {
            self.push(v);
        }
    }

    /// Finish as the smallest encoding: inline (FI) when it beats the
    /// standard layout, else standard. Frozen bitmaps are built for storage,
    /// so the builder always finishes compact.
    pub fn finish(mut self) -> FrozenBitmap {
        if self.have_cur {
            self.flush();
        }
        if self.total as usize <= INLINE_MAX_COUNT {
            let (standard_total, ..) = layout(&self.containers);
            if inline_size(self.total as usize) < standard_total {
                return serialize_inline(&self.containers, self.total as usize);
            }
        }
        serialize_standard(&self.containers, self.total)
    }

    /// Finish as standard format unconditionally (op-ready, never inline).
    /// Exposed under `internals` for white-box tests and benchmarks.
    #[cfg(feature = "internals")]
    pub fn finish_standard(mut self) -> FrozenBitmap {
        if self.have_cur {
            self.flush();
        }
        serialize_standard(&self.containers, self.total)
    }

    fn flush(&mut self) {
        let built = build_container(self.cur_key, &self.cur);
        self.total += built.card as u64;
        self.containers.push(built);
        self.cur.clear();
    }
}

impl Default for FrozenBitmapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The empty bitmap, without constructing a builder: what `finish` yields for one
/// that was never pushed to, minus the eagerly reserved accumulator.
pub(crate) fn empty() -> FrozenBitmap {
    serialize_inline(&[], 0)
}

/// Pick the smallest representation for one key's sorted lows.
fn build_container(key: u16, vals: &[u16]) -> Built {
    let card = vals.len() as u32;
    let run_count = count_runs(vals);

    let array_cost = vals.len() * 2;
    let run_cost = run_bytes(run_count);
    let bitmap_cost = BITMAP_BYTES;

    let (typ, data) = if run_cost <= array_cost && run_cost <= bitmap_cost {
        (CT_RUN, ContainerData::Run(extract_runs(vals, run_count)))
    } else if array_cost <= bitmap_cost {
        (CT_ARRAY, ContainerData::Array(vals.to_vec()))
    } else {
        let mut words = Box::new([0u64; BITMAP_WORDS]);
        for &v in vals {
            words[v as usize / 64] |= 1u64 << (v as usize % 64);
        }
        (CT_BITMAP, ContainerData::Bitmap(words))
    };

    Built {
        key,
        typ,
        card,
        data,
    }
}

/// Number of maximal runs of consecutive values in sorted, deduped `vals`.
fn count_runs(vals: &[u16]) -> usize {
    let pairs = vals.iter().zip(vals.iter().skip(1));
    usize::from(!vals.is_empty()) + pairs.filter(|(&a, &b)| b != a + 1).count()
}

/// Run-length encode sorted, deduped lows into runs covering the inclusive
/// range `[start, start + len]`. `run_count` must be [`count_runs`] of `vals`,
/// so the result is allocated once at exact size.
fn extract_runs(vals: &[u16], run_count: usize) -> Vec<Run> {
    let mut runs = Vec::with_capacity(run_count);
    let mut start = vals[0];
    let mut prev = vals[0];
    for &v in &vals[1..] {
        if v == prev + 1 {
            prev = v;
        } else {
            let len = prev - start;
            runs.push(Run { start, len });
            start = v;
            prev = v;
        }
    }
    let len = prev - start;
    runs.push(Run { start, len });
    runs
}

/// Standard-layout plan: total size and flags (bitmaps 64-aligned, the rest
/// 2-aligned). `serialize_standard` repeats the walk for the offsets, so nothing
/// here is allocated.
fn layout(containers: &[Built]) -> (usize, bool, bool) {
    let has_runs = containers.iter().any(|c| c.typ == CT_RUN);
    let has_bitmap = containers.iter().any(|c| c.typ == CT_BITMAP);
    let mut cursor = 0usize;
    for c in containers {
        let align = if c.typ == CT_BITMAP { BUF_ALIGN } else { 2 };
        cursor = align_up(cursor, align) + c.data.byte_len();
    }
    let total = data_section_off(containers.len(), has_bitmap) + cursor;
    (total, has_runs, has_bitmap)
}

/// Lay out header + SoA index + data section into a 64-aligned buffer.
fn serialize_standard(containers: &[Built], total_card: u64) -> FrozenBitmap {
    let n = containers.len();
    let (total, has_runs, has_bitmap) = layout(containers);
    let data_base = data_section_off(n, has_bitmap);

    let mut buf = result_buf(total);
    buf.resize(total, 0);

    Header {
        has_runs,
        has_bitmap,
        num_containers: n as u32,
        cardinality: total_card,
    }
    .write(&mut buf);

    let mut cursor = 0usize;
    for (i, c) in containers.iter().enumerate() {
        let align = if c.typ == CT_BITMAP { BUF_ALIGN } else { 2 };
        cursor = align_up(cursor, align);
        write_index_entry(
            &mut buf,
            n,
            i,
            IndexEntry {
                key: c.key,
                typ: c.typ,
                cardinality: c.card,
                data_offset: cursor as u32,
            },
        );
        let start = data_base + cursor;
        let len = c.data.byte_len();
        c.data.write(&mut buf[start..start + len]);
        cursor += len;
    }

    FrozenBitmap::from_buf(buf)
}

/// Re-expand built containers into packed u32s ("FI" + u16 count + values).
/// Only reached when FI won the size comparison, i.e. tiny per-key counts.
fn serialize_inline(containers: &[Built], count: usize) -> FrozenBitmap {
    let total = inline_size(count);
    let mut buf = result_buf(total);
    buf.resize(total, 0);
    buf[0..2].copy_from_slice(&INLINE_MAGIC);
    write_u16(&mut buf, INLINE_COUNT_OFF, count as u16);

    let mut off = INLINE_HEADER_SIZE;
    for c in containers {
        let hi = (c.key as u32) << 16;
        c.data.view().for_each(|lo| {
            write_u32(&mut buf, off, hi | lo as u32);
            off += 4;
        });
    }
    debug_assert_eq!(off, INLINE_HEADER_SIZE + 4 * count);

    FrozenBitmap::from_buf(buf)
}
