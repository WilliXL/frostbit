//! Wire format: constants and byte-level primitives. Little-endian only.
//!
//! Standard format (`FROZ`, v3): 16-byte header, then a structure-of-arrays
//! container index, then the data section.
//!
//! ```text
//!   0  u32  MAGIC          ("FROZ")
//!   4  u16  VERSION        (3)
//!   6  u16  FLAGS          (FLAG_HAS_RUNS | FLAG_FULL | …)
//!   8  u32  NUM_CONTAINERS
//!  12  u32  CARDINALITY    (FLAG_FULL ⇒ true value is 2^32)
//!  16       index (SoA) + data section
//! ```

#[cfg(target_endian = "big")]
compile_error!("frostbit assumes a little-endian target");

// --- Standard format: magic & versions -------------------------------------

/// Standard-format magic. Reads "FROZ" in big-endian hex; matches the jata
/// reference so a reader can dispatch on shared magic + [`VERSION`].
pub const MAGIC: u32 = 0x46524F5A;

/// Version frostbit writes (compact header + SoA index).
pub const VERSION: u16 = 3;

// --- Header field offsets (16-byte header) ---------------------------------

pub const H_MAGIC: usize = 0;
pub const H_VERSION: usize = 4;
pub const H_FLAGS: usize = 6;
pub const H_NUM_CONTAINERS: usize = 8;
pub const H_CARDINALITY: usize = 12;
pub const HEADER_SIZE: usize = 16;

// --- Header flags ----------------------------------------------------------

/// At least one container is run-encoded.
pub const FLAG_HAS_RUNS: u16 = 1 << 0;
/// Bitmap holds every value in `0..=u32::MAX`; the `u32` field can't store `2^32`.
pub const FLAG_FULL: u16 = 1 << 1;
/// At least one container is a bitmap; the data section is then 64-aligned.
pub const FLAG_HAS_BITMAP: u16 = 1 << 2;

// --- Container types -------------------------------------------------------

pub const CT_ARRAY: u8 = 0;
pub const CT_BITMAP: u8 = 1;
pub const CT_RUN: u8 = 2;
/// Not a stored type: a cursor marker for an inline-format key group, whose
/// payload is packed `u32`s (treated as an array of `card` lows for sizing).
pub const CT_INLINE: u8 = 3;

// --- Container sizing ------------------------------------------------------

/// A bitmap container covers a 2^16 key space as 1024 × `u64`.
pub const BITMAP_WORDS: usize = 1024;
/// Bitmap payload size in bytes.
pub const BITMAP_BYTES: usize = BITMAP_WORDS * 8;
/// Array beats bitmap (2 bytes/value vs fixed 8 KiB) up to this cardinality.
pub const ARRAY_MAX_SIZE: usize = 4096;
/// Max runs that fit a run payload before it's larger than a bitmap.
pub const MAX_RUNS: usize = (BITMAP_BYTES - 2) / 4;

/// Stored size of a run container: a `u16` count then `nr` `(start, len)` pairs.
#[inline(always)]
pub const fn run_bytes(nr: usize) -> usize {
    2 + nr * 4
}

const _: () = assert!(BITMAP_BYTES == 8192);

// --- Alignment -------------------------------------------------------------

/// `FrozenBitmap` allocation + bitmap-container alignment, so 512-bit SIMD
/// loads never split a cache line.
pub const BUF_ALIGN: usize = 64;
/// Baseline section alignment (`u64`-castable).
pub const WORD_ALIGN: usize = 8;

/// Round `n` up to a multiple of `align` (a power of two).
#[inline(always)]
pub const fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

// --- Little-endian readers / writers ---------------------------------------

#[inline(always)]
pub fn read_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

#[inline(always)]
pub fn read_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

#[inline(always)]
pub fn read_u64(bytes: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
        bytes[off + 4],
        bytes[off + 5],
        bytes[off + 6],
        bytes[off + 7],
    ])
}

#[inline(always)]
pub fn write_u16(bytes: &mut [u8], off: usize, val: u16) {
    bytes[off..off + 2].copy_from_slice(&val.to_le_bytes());
}

#[inline(always)]
pub fn write_u32(bytes: &mut [u8], off: usize, val: u32) {
    bytes[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline(always)]
pub fn write_u64(bytes: &mut [u8], off: usize, val: u64) {
    bytes[off..off + 8].copy_from_slice(&val.to_le_bytes());
}

// --- Inline format (FI) ----------------------------------------------------
//
// Whole-bitmap encoding for scattered/small sets: [0..2) magic "FI",
// [2..4) u16 count, [4..) packed ascending u32 values. Total align8(4 + 4n).

/// Inline-format magic bytes.
pub const INLINE_MAGIC: [u8; 2] = *b"FI";
/// Header: magic (2) + u16 count (2); values start here, 4-byte aligned.
pub const INLINE_HEADER_SIZE: usize = 4;
/// Byte offset of the u16 value count.
pub const INLINE_COUNT_OFF: usize = 2;
/// Max values an inline bitmap can hold (u16 count).
pub const INLINE_MAX_COUNT: usize = u16::MAX as usize;

/// Serialized size of an inline bitmap with `n` values.
#[inline(always)]
pub const fn inline_size(n: usize) -> usize {
    align_up(INLINE_HEADER_SIZE + 4 * n, WORD_ALIGN)
}

/// Whether `bytes` starts with the inline magic.
#[inline(always)]
pub fn has_inline_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0..2] == INLINE_MAGIC
}

// --- Header codec ----------------------------------------------------------

/// Cardinality of a full bitmap (`2^32`); carried by [`FLAG_FULL`].
pub const FULL_CARDINALITY: u64 = 1 << 32;

/// Decoded standard-format header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub has_runs: bool,
    pub has_bitmap: bool,
    pub num_containers: u32,
    /// Total set bits, `0..=FULL_CARDINALITY`.
    pub cardinality: u64,
}

impl Header {
    /// Write the 16-byte header into the front of `buf`.
    #[inline]
    pub fn write(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= HEADER_SIZE);
        debug_assert!(self.cardinality <= FULL_CARDINALITY);

        let mut flags = 0u16;
        if self.has_runs {
            flags |= FLAG_HAS_RUNS;
        }
        if self.has_bitmap {
            flags |= FLAG_HAS_BITMAP;
        }
        let card_field = if self.cardinality == FULL_CARDINALITY {
            flags |= FLAG_FULL;
            0
        } else {
            self.cardinality as u32
        };

        write_u32(buf, H_MAGIC, MAGIC);
        write_u16(buf, H_VERSION, VERSION);
        write_u16(buf, H_FLAGS, flags);
        write_u32(buf, H_NUM_CONTAINERS, self.num_containers);
        write_u32(buf, H_CARDINALITY, card_field);
    }

    /// Parse a v3 header; `None` on short input, bad magic, or wrong version.
    #[inline]
    pub fn parse(bytes: &[u8]) -> Option<Header> {
        if bytes.len() < HEADER_SIZE {
            return None;
        }
        if read_u32(bytes, H_MAGIC) != MAGIC {
            return None;
        }
        if read_u16(bytes, H_VERSION) != VERSION {
            return None;
        }
        let flags = read_u16(bytes, H_FLAGS);
        let card_field = read_u32(bytes, H_CARDINALITY);
        Some(Header {
            has_runs: flags & FLAG_HAS_RUNS != 0,
            has_bitmap: flags & FLAG_HAS_BITMAP != 0,
            num_containers: read_u32(bytes, H_NUM_CONTAINERS),
            cardinality: if flags & FLAG_FULL != 0 {
                FULL_CARDINALITY
            } else {
                card_field as u64
            },
        })
    }
}

// --- Container index (structure-of-arrays) ---------------------------------
//
// keys[u16;n] @16, cards[u16;n] @16+2n (card-1), type_offsets[u32;n] @16+4n
// (2-bit type << 30 | 30-bit offset). Key search streams only `keys`.

/// Bit position of the 2-bit container type in a `type_offset` word.
pub const E_TYPE_SHIFT: u32 = 30;
/// Mask for the 30-bit data offset (max ~1 GiB).
pub const E_OFFSET_MASK: u32 = (1 << E_TYPE_SHIFT) - 1;

/// Index byte size for `n` containers.
#[inline(always)]
pub const fn index_size(n: usize) -> usize {
    8 * n
}

/// Data section start: 64-aligned iff a bitmap container exists, else 8.
#[inline(always)]
pub const fn data_section_off(n: usize, has_bitmap: bool) -> usize {
    align_up(
        HEADER_SIZE + index_size(n),
        if has_bitmap { BUF_ALIGN } else { WORD_ALIGN },
    )
}

#[inline(always)]
pub const fn keys_off() -> usize {
    HEADER_SIZE
}

#[inline(always)]
pub const fn cards_off(n: usize) -> usize {
    HEADER_SIZE + 2 * n
}

#[inline(always)]
pub const fn type_offsets_off(n: usize) -> usize {
    HEADER_SIZE + 4 * n
}

/// A decoded container index slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    pub key: u16,
    pub typ: u8,
    /// Cardinality, 1..=65536.
    pub cardinality: u32,
    /// Payload offset from the start of the data section.
    pub data_offset: u32,
}

/// The container index borrowed as its three typed sub-arrays.
///
/// The SoA layout exists so a walk touches one field at a time; reading it as
/// real slices means each visit is one load per field instead of assembling
/// bytes. Every sub-array is correctly aligned for its element type whenever the
/// buffer base is (`keys` at +16, `cards` 2-aligned, `type_offsets` 4-aligned),
/// which is checked when untrusted bytes are parsed.
#[derive(Clone, Copy)]
pub struct Index<'a> {
    keys: &'a [u16],
    cards: &'a [u16],
    type_offsets: &'a [u32],
}

impl<'a> Index<'a> {
    /// Borrow the index of an `n`-container standard bitmap.
    ///
    /// `bytes` must start at a [`WORD_ALIGN`]-aligned address — the index is
    /// reinterpreted as `u16`/`u32` slices zero-copy, and the format places
    /// them at aligned offsets *from the base*. Both `from_bytes` boundaries
    /// reject under-aligned buffers before an `Index` can exist, so inside the
    /// crate this holds by construction; the assert catches a caller (a test,
    /// an `internals` user) handing in a bare `Vec<u8>`, which the allocator
    /// does not align on every platform Miri models.
    #[inline]
    pub fn new(bytes: &'a [u8], n: usize) -> Self {
        debug_assert!(
            (bytes.as_ptr() as usize).is_multiple_of(WORD_ALIGN),
            "index base must be WORD_ALIGN-aligned (from_bytes enforces this)"
        );
        Index {
            keys: bytemuck::cast_slice(&bytes[keys_off()..keys_off() + 2 * n]),
            cards: bytemuck::cast_slice(&bytes[cards_off(n)..cards_off(n) + 2 * n]),
            type_offsets: bytemuck::cast_slice(
                &bytes[type_offsets_off(n)..type_offsets_off(n) + 4 * n],
            ),
        }
    }

    /// Key of container `i`, or `None` past the end.
    #[inline]
    pub fn key(&self, i: usize) -> Option<u16> {
        self.keys.get(i).copied()
    }

    /// Position of the container holding `key`, or `None` if absent.
    #[inline]
    pub fn find(&self, key: u16) -> Option<usize> {
        self.keys.binary_search(&key).ok()
    }

    /// Decoded entry `i`.
    #[inline]
    pub fn entry(&self, i: usize) -> IndexEntry {
        let to = self.type_offsets[i];
        IndexEntry {
            key: self.keys[i],
            typ: (to >> E_TYPE_SHIFT) as u8,
            cardinality: self.cards[i] as u32 + 1,
            data_offset: to & E_OFFSET_MASK,
        }
    }
}

#[inline]
pub fn read_index_entry(bytes: &[u8], n: usize, i: usize) -> IndexEntry {
    debug_assert!(i < n);
    let to = read_u32(bytes, type_offsets_off(n) + i * 4);
    IndexEntry {
        key: read_u16(bytes, keys_off() + i * 2),
        typ: (to >> E_TYPE_SHIFT) as u8,
        cardinality: read_u16(bytes, cards_off(n) + i * 2) as u32 + 1,
        data_offset: to & E_OFFSET_MASK,
    }
}

#[inline]
pub fn write_index_entry(buf: &mut [u8], n: usize, i: usize, e: IndexEntry) {
    debug_assert!(i < n);
    debug_assert!(e.cardinality >= 1 && e.cardinality <= 65_536);
    debug_assert!(e.data_offset <= E_OFFSET_MASK);
    debug_assert!((e.typ as u32) <= 0b11);
    write_u16(buf, keys_off() + i * 2, e.key);
    write_u16(buf, cards_off(n) + i * 2, (e.cardinality - 1) as u16);
    let to = ((e.typ as u32) << E_TYPE_SHIFT) | (e.data_offset & E_OFFSET_MASK);
    write_u32(buf, type_offsets_off(n) + i * 4, to);
}

