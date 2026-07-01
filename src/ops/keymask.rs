//! Live container-key set for hole-punching.
//!
//! A container key is the high 16 bits of a value — which 64K block it lives in.
//! In `AND(x, OR(a, b, …))`, a key absent from `x` is dead in the result, so the
//! `OR` need never materialize it. [`KeyMask`] is the set of keys that can
//! survive to the root; cursors skip every block whose key isn't in it, pruning
//! dead work before any fold. Pruning is by key only — monotone and
//! result-preserving (cardinality is identical).

/// A dense bitset over the whole 2¹⁶ key space (8 KiB). Dense, not a sorted vec,
/// because the hot path is an O(1) per-container `contains` probe.
pub struct KeyMask {
    words: Box<[u64; Self::WORDS]>,
}

impl KeyMask {
    const WORDS: usize = 1 << 10; // 1024 × u64 = 2^16 bits

    #[inline]
    pub fn empty() -> Self {
        KeyMask { words: Box::new([0u64; Self::WORDS]) }
    }

    #[inline]
    pub fn set(&mut self, key: u16) {
        self.words[key as usize >> 6] |= 1u64 << (key & 63);
    }

    #[inline]
    pub fn contains(&self, key: u16) -> bool {
        (self.words[key as usize >> 6] >> (key & 63)) & 1 == 1
    }

    /// In-place intersection (AND of key sets).
    pub fn intersect_with(&mut self, other: &KeyMask) {
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= *b;
        }
    }

    /// In-place union (OR of key sets).
    pub fn union_with(&mut self, other: &KeyMask) {
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
    }
}
