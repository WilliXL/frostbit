//! Byte-shuffle table that compacts the selected lanes of an 8-lane block.

/// Byte-shuffle indices that compact the set lanes of a `u16x8` to the front,
/// keyed by the 8-bit lane mask. Unused trailing bytes are `0xFF` (≥16), which
/// both NEON `vqtbl1q` and x86 `pshufb` map to zero. `popcount(mask)` lanes valid.
pub(crate) const COMPACT: [[u8; 16]; 256] = {
    let mut t = [[0xFFu8; 16]; 256];
    let mut m = 0usize;
    while m < 256 {
        let (mut pos, mut lane) = (0usize, 0usize);
        while lane < 8 {
            if m & (1 << lane) != 0 {
                t[m][2 * pos] = (2 * lane) as u8;
                t[m][2 * pos + 1] = (2 * lane + 1) as u8;
                pos += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    t
};
