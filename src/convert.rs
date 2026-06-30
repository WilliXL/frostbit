//! Conversions to/from `roaring::RoaringBitmap`.

use roaring::RoaringBitmap;

use crate::{FrozenBitmap, FrozenBitmapBuilder, FrozenBitmapView};

impl FrozenBitmap {
    /// Compact frozen bitmap holding the same values as `rb`.
    pub fn from_roaring(rb: &RoaringBitmap) -> Self {
        let mut b = FrozenBitmapBuilder::new();
        b.extend_sorted(rb.iter());
        b.finish()
    }

    /// Materialize as a mutable roaring bitmap.
    pub fn to_roaring(&self) -> RoaringBitmap {
        self.view().to_roaring()
    }
}

impl FrozenBitmapView<'_> {
    /// Materialize as a mutable roaring bitmap.
    pub fn to_roaring(&self) -> RoaringBitmap {
        RoaringBitmap::from_sorted_iter(self.iter()).expect("frozen iteration is strictly ascending")
    }
}

impl From<&RoaringBitmap> for FrozenBitmap {
    fn from(rb: &RoaringBitmap) -> Self {
        Self::from_roaring(rb)
    }
}

impl From<&FrozenBitmap> for RoaringBitmap {
    fn from(bm: &FrozenBitmap) -> Self {
        bm.to_roaring()
    }
}
