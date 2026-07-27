//! Op kernel tests (intersect / union / diff), differential against `roaring`.
//! Runs in debug, so the arena's no-alloc `record` debug-assert fires on any
//! slot overflow.
#![cfg(feature = "internals")]

mod common;
mod difference;
mod intersect;
mod union;
