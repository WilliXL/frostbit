//! Builds a frozen bitmap from values pushed in strictly ascending order.

use crate::bitmap::{result_buf, FrozenBitmap};
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
    cur: Vec<u16>,
    have_cur: bool,
    total: u64,
}

struct Built {
    key: u16,
    typ: u8,
    card: u32,
    payload: Vec<u8>,
}

impl FrozenBitmapBuilder {
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
            cur_key: 0,
            cur: Vec::new(),
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

    pub fn extend_sorted(&mut self, iter: impl IntoIterator<Item = u32>) {
        for v in iter {
            self.push(v);
        }
    }

    /// Finish as the smallest encoding: inline (FRI) when it beats the
    /// standard layout, else standard. Frozen bitmaps are built for storage,
    /// so the builder always finishes compact.
    pub fn finish(mut self) -> FrozenBitmap {
        if self.have_cur {
            self.flush();
        }
        if self.total as usize <= INLINE_MAX_COUNT {
            let (_, standard_total, ..) = layout(&self.containers);
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
        let vals = std::mem::take(&mut self.cur);
        let built = build_container(self.cur_key, &vals);
        self.total += built.card as u64;
        self.containers.push(built);
    }
}

impl Default for FrozenBitmapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Pick the smallest representation for one key's sorted lows and serialize it.
fn build_container(key: u16, vals: &[u16]) -> Built {
    let card = vals.len() as u32;
    let runs = extract_runs(vals);

    let array_cost = vals.len() * 2;
    let run_cost = 2 + runs.len() * 4;
    let bitmap_cost = BITMAP_BYTES;

    let (typ, payload) = if run_cost <= array_cost && run_cost <= bitmap_cost {
        let mut p = vec![0u8; run_cost];
        write_u16(&mut p, 0, runs.len() as u16);
        for (j, &(start, len)) in runs.iter().enumerate() {
            write_u16(&mut p, 2 + j * 4, start);
            write_u16(&mut p, 2 + j * 4 + 2, len);
        }
        (CT_RUN, p)
    } else if array_cost <= bitmap_cost {
        let mut p = vec![0u8; array_cost];
        for (j, &v) in vals.iter().enumerate() {
            write_u16(&mut p, j * 2, v);
        }
        (CT_ARRAY, p)
    } else {
        let mut words = [0u64; BITMAP_WORDS];
        for &v in vals {
            words[v as usize / 64] |= 1u64 << (v as usize % 64);
        }
        let mut p = vec![0u8; BITMAP_BYTES];
        for (j, &w) in words.iter().enumerate() {
            write_u64(&mut p, j * 8, w);
        }
        (CT_BITMAP, p)
    };

    Built {
        key,
        typ,
        card,
        payload,
    }
}

/// Run-length encode sorted, deduped lows into `(start, length)` pairs, where a
/// pair covers the inclusive range `[start, start + length]`.
fn extract_runs(vals: &[u16]) -> Vec<(u16, u16)> {
    let mut runs = Vec::new();
    let mut start = vals[0];
    let mut prev = vals[0];
    for &v in &vals[1..] {
        if v == prev + 1 {
            prev = v;
        } else {
            runs.push((start, prev - start));
            start = v;
            prev = v;
        }
    }
    runs.push((start, prev - start));
    runs
}

/// Standard-layout plan: per-container payload offsets (bitmaps 64-aligned,
/// the rest 2-aligned), total size, and flags.
fn layout(containers: &[Built]) -> (Vec<u32>, usize, bool, bool) {
    let has_runs = containers.iter().any(|c| c.typ == CT_RUN);
    let has_bitmap = containers.iter().any(|c| c.typ == CT_BITMAP);
    let mut offsets = Vec::with_capacity(containers.len());
    let mut cursor = 0usize;
    for c in containers {
        let align = if c.typ == CT_BITMAP { BUF_ALIGN } else { 2 };
        cursor = align_up(cursor, align);
        offsets.push(cursor as u32);
        cursor += c.payload.len();
    }
    let total = data_section_off(containers.len(), has_bitmap) + cursor;
    (offsets, total, has_runs, has_bitmap)
}

/// Lay out header + SoA index + data section into a 64-aligned buffer.
fn serialize_standard(containers: &[Built], total_card: u64) -> FrozenBitmap {
    let n = containers.len();
    let (offsets, total, has_runs, has_bitmap) = layout(containers);
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

    for (i, c) in containers.iter().enumerate() {
        write_index_entry(
            &mut buf,
            n,
            i,
            IndexEntry {
                key: c.key,
                typ: c.typ,
                cardinality: c.card,
                data_offset: offsets[i],
            },
        );
    }

    for (i, c) in containers.iter().enumerate() {
        let start = data_base + offsets[i] as usize;
        buf[start..start + c.payload.len()].copy_from_slice(&c.payload);
    }

    FrozenBitmap::from_buf(buf)
}

/// Re-expand built containers into packed u32s ("FI" + u16 count + values).
/// Only reached when FRI won the size comparison, i.e. tiny per-key counts.
fn serialize_inline(containers: &[Built], count: usize) -> FrozenBitmap {
    let total = inline_size(count);
    let mut buf = result_buf(total);
    buf.resize(total, 0);
    buf[0..2].copy_from_slice(&INLINE_MAGIC);
    write_u16(&mut buf, INLINE_COUNT_OFF, count as u16);

    let mut off = INLINE_HEADER_SIZE;
    for c in containers {
        let hi = (c.key as u32) << 16;
        match c.typ {
            CT_ARRAY => {
                for j in 0..c.card as usize {
                    write_u32(&mut buf, off, hi | read_u16(&c.payload, j * 2) as u32);
                    off += 4;
                }
            }
            CT_RUN => {
                let nr = read_u16(&c.payload, 0) as usize;
                for r in 0..nr {
                    let s = read_u16(&c.payload, 2 + r * 4) as u32;
                    let e = s + read_u16(&c.payload, 2 + r * 4 + 2) as u32;
                    for v in s..=e {
                        write_u32(&mut buf, off, hi | v);
                        off += 4;
                    }
                }
            }
            _ => {
                for w in 0..BITMAP_WORDS {
                    let mut word = read_u64(&c.payload, w * 8);
                    while word != 0 {
                        let tz = word.trailing_zeros() as usize;
                        write_u32(&mut buf, off, hi | (w * 64 + tz) as u32);
                        word &= word - 1;
                        off += 4;
                    }
                }
            }
        }
    }
    debug_assert_eq!(off, INLINE_HEADER_SIZE + 4 * count);

    FrozenBitmap::from_buf(buf)
}
