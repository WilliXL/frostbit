//! Working-memory pools: budgeting, pre-allocation, and overflow policy.
//!
//! Every op reuses its working memory from a small per-thread pool, so a warm
//! `intersect_fast` / `materialize` performs **zero allocations**. That is
//! automatic and needs no setup — this module exists to put an explicit *upper
//! bound* on that memory, to pre-allocate it up front, and to say what should
//! happen if a fold ever needs more than the budget.
//!
//! Pools are per-thread, so there is no contention and no shared state to
//! synchronize: each worker thread owns its own budget. To use frostbit under a
//! thread pool, configure and pre-allocate in the pool's thread-start hook:
//!
//! ```ignore
//! use frostbit::pool::{self, PoolConfig};
//!
//! let warm = || {
//!     pool::configure(PoolConfig::new().buffers(8).buffer_bytes(1 << 20));
//!     pool::prewarm();
//! };
//!
//! // rayon: once per worker thread
//! rayon::ThreadPoolBuilder::new().start_handler(move |_| warm()).build()?;
//!
//! // tokio: once per runtime worker
//! tokio::runtime::Builder::new_multi_thread().on_thread_start(warm).build()?;
//! ```
//!
//! Because the budget is per-thread, total working memory is bounded by
//! `threads × budget`.

use std::cell::{Cell, RefCell};

/// What happens when a fold needs a buffer and the budget is already fully
/// handed out (e.g. an expression tree with more live intermediates than
/// [`PoolConfig::buffers`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnOverflow {
    /// Allocate the extra buffer on demand and **drop** it when the fold
    /// releases it, so the pool never grows past its budget. Ops always
    /// succeed; the budget bounds *retained* memory, not peak memory.
    /// This is the default.
    #[default]
    Allocate,
    /// Panic instead, naming the budget that was exceeded.
    ///
    /// Use this as a **budget assertion** in tests and CI to catch a workload
    /// that outgrew its pre-allocation; prefer [`OnOverflow::Allocate`] in
    /// production, where degrading beats crashing.
    Fail,
}

/// How much working memory a thread may keep, and what to do if a fold needs
/// more. Build with the chained setters, then apply with [`configure`].
///
/// ```
/// use frostbit::pool::{self, OnOverflow, PoolConfig};
///
/// // 16 buffers, pre-sized 1 MiB each.
/// pool::configure(PoolConfig::new().buffers(16).buffer_bytes(1 << 20));
///
/// // Or an explicit shape — one entry per buffer.
/// pool::configure(PoolConfig::new().buffer_sizes([4 << 20, 1 << 20, 1 << 20]));
///
/// // Treat exceeding the budget as a bug rather than allocating.
/// pool::configure(PoolConfig::new().buffers(4).on_overflow(OnOverflow::Fail));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolConfig {
    buffers: usize,
    sizes: Vec<usize>,
    on_overflow: OnOverflow,
}

/// Buffers retained per pool when nothing is configured.
pub(crate) const DEFAULT_BUFFERS: usize = 8;

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig { buffers: DEFAULT_BUFFERS, sizes: Vec::new(), on_overflow: OnOverflow::default() }
    }
}

impl PoolConfig {
    /// The default budget: 8 buffers, allocated on demand, with
    /// [`OnOverflow::Allocate`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep at most `n` buffers live at once (per pool, per thread).
    pub fn buffers(mut self, n: usize) -> Self {
        self.buffers = n;
        self.sizes.truncate(n);
        self
    }

    /// Pre-size every buffer to `bytes` (see [`prewarm`]).
    pub fn buffer_bytes(mut self, bytes: usize) -> Self {
        self.sizes = vec![bytes; self.buffers];
        self
    }

    /// Pre-size buffers individually — one entry per buffer, for a workload
    /// whose folds have known, uneven working sets. Also sets the buffer count.
    pub fn buffer_sizes(mut self, sizes: impl IntoIterator<Item = usize>) -> Self {
        self.sizes = sizes.into_iter().collect();
        self.buffers = self.sizes.len();
        self
    }

    /// What to do when a fold needs more buffers than the budget.
    pub fn on_overflow(mut self, policy: OnOverflow) -> Self {
        self.on_overflow = policy;
        self
    }

    /// Bytes [`prewarm`] would allocate under this config.
    ///
    /// The sizes apply to each of the two byte pools — arena working memory and
    /// result buffers — because a result is serialized *in place* in its arena
    /// and the two pools swap buffers on every op. So this is twice the
    /// configured sum, and warming only one of them would leave the other cold.
    pub fn prewarm_bytes(&self) -> usize {
        2 * self.sizes.iter().sum::<usize>()
    }
}

/// A snapshot of this thread's byte-buffer pools (arena working memory and
/// result buffers) — the memory that actually scales with bitmap size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    /// Buffers handed out right now — in-flight fold working memory, plus the
    /// buffer behind every [`FrozenBitmap`](crate::FrozenBitmap) you still hold
    /// (a result owns its buffer until it drops).
    pub live: usize,
    /// Idle buffers held for reuse.
    pub retained: usize,
    /// Bytes of capacity held by idle buffers.
    pub retained_bytes: usize,
    /// Times a request exceeded the budget since the last [`configure`].
    pub overflows: u64,
}

/// The hot-path slice of the config: `Copy`, so `take`/`put` read it with a
/// plain load instead of touching the `Vec` of sizes.
#[derive(Clone, Copy)]
pub(crate) struct Budget {
    pub buffers: usize,
    pub on_overflow: OnOverflow,
}

thread_local! {
    static BUDGET: Cell<Budget> =
        const { Cell::new(Budget { buffers: DEFAULT_BUFFERS, on_overflow: OnOverflow::Allocate }) };
    static SIZES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static OVERFLOWS: Cell<u64> = const { Cell::new(0) };
}

#[inline]
pub(crate) fn budget() -> Budget {
    BUDGET.with(Cell::get)
}

/// Record an over-budget request, and panic under [`OnOverflow::Fail`].
#[cold]
pub(crate) fn overflow(pool: &str, live: usize, limit: usize) {
    OVERFLOWS.with(|c| c.set(c.get().saturating_add(1)));
    if budget().on_overflow == OnOverflow::Fail {
        panic!(
            "frostbit: {pool} pool budget exceeded — {live} buffers live, budget is {limit}. \
             Raise PoolConfig::buffers, or use OnOverflow::Allocate to allocate on demand."
        );
    }
}

/// Apply `config` to the **current thread**. Existing idle buffers beyond the
/// new budget are released on their next return; call [`prewarm`] to allocate
/// the new budget up front, or [`clear`] to drop what is held now.
pub fn configure(config: PoolConfig) {
    BUDGET.with(|b| b.set(Budget { buffers: config.buffers, on_overflow: config.on_overflow }));
    SIZES.with(|s| *s.borrow_mut() = config.sizes);
    OVERFLOWS.with(|c| c.set(0));
}

/// The current thread's config-visible budget.
pub fn config() -> PoolConfig {
    let b = budget();
    PoolConfig {
        buffers: b.buffers,
        sizes: SIZES.with(|s| s.borrow().clone()),
        on_overflow: b.on_overflow,
    }
}

/// Allocate this thread's configured buffers now, so the first fold does not
/// pay for them. Without a size (`buffer_bytes` / `buffer_sizes`) there is
/// nothing to pre-size and this is a no-op.
pub fn prewarm() {
    let sizes = SIZES.with(|s| s.borrow().clone());
    if sizes.is_empty() {
        return;
    }
    crate::api::bitmap::prewarm_result_pool(&sizes);
    crate::ops::arena::prewarm_arena_pool(&sizes);
}

/// Release every idle buffer this thread holds. Call this when a worker goes
/// idle and you would rather hand the memory back than keep it warm.
///
/// The budget is unchanged, so buffers still in flight are pooled again when
/// their fold releases them, and later folds re-warm as usual. To stop pooling
/// entirely, configure `buffers(0)`.
pub fn clear() {
    crate::api::bitmap::clear_result_pool();
    crate::ops::arena::clear_arena_pool();
    crate::api::expr::clear_stack_pool();
    crate::ops::cursor::clear_scratch_pool();
    crate::ops::analyze::plan::clear_slot_pool();
}

/// A snapshot of this thread's byte-buffer pools.
pub fn stats() -> PoolStats {
    let (result_live, result_retained, result_bytes) = crate::api::bitmap::result_pool_stats();
    let (arena_live, arena_retained, arena_bytes) = crate::ops::arena::arena_pool_stats();
    PoolStats {
        live: result_live + arena_live,
        retained: result_retained + arena_retained,
        retained_bytes: result_bytes + arena_bytes,
        overflows: OVERFLOWS.with(Cell::get),
    }
}

/// A per-thread free list with a budget: the shared machinery behind every
/// frostbit pool (arena working memory, result buffers, and the small
/// bookkeeping vectors).
///
/// `take` hands out an idle item or makes one; `put` retains it while the pool
/// is under budget and drops it otherwise, so retained memory never exceeds the
/// configured bound. `live` counts what is currently handed out, which is what
/// [`OnOverflow`] is judged against.
pub(crate) struct Pool<T> {
    free: RefCell<Vec<T>>,
    live: Cell<usize>,
    name: &'static str,
    kind: Kind,
}

/// Whether exceeding the budget is a fold outgrowing its working set (which
/// [`OnOverflow`] judges) or simply more results outstanding than the recycling
/// cache holds (which is the caller's business, not an overflow).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Per-fold working memory.
    Working,
    /// Buffers handed to values the caller owns, recycled when they drop.
    Cache,
}

impl<T> Pool<T> {
    /// A pool of per-fold working memory, subject to [`OnOverflow`].
    pub(crate) const fn new(name: &'static str) -> Self {
        Pool { free: RefCell::new(Vec::new()), live: Cell::new(0), name, kind: Kind::Working }
    }

    /// A recycling cache for buffers the caller ends up owning. Retention is
    /// still bounded by the budget, but holding many results is not an overflow.
    pub(crate) const fn cache(name: &'static str) -> Self {
        Pool { free: RefCell::new(Vec::new()), live: Cell::new(0), name, kind: Kind::Cache }
    }

    /// An idle item, or a fresh one — reporting an overflow if a fold's working
    /// set is already fully handed out.
    #[inline]
    pub(crate) fn take(&self, make: impl FnOnce() -> T) -> T {
        self.live.set(self.live.get() + 1);
        if let Some(item) = self.free.borrow_mut().pop() {
            return item;
        }
        let limit = budget().buffers;
        if self.kind == Kind::Working && self.live.get() > limit {
            overflow(self.name, self.live.get(), limit);
        }
        make()
    }

    /// Return an item, retaining it only while the pool is under budget.
    #[inline]
    pub(crate) fn put(&self, item: T) {
        self.live.set(self.live.get().saturating_sub(1));
        let mut free = self.free.borrow_mut();
        if free.len() < budget().buffers {
            free.push(item);
        }
    }

    /// Pre-allocate one item per entry of `sizes`, up to the budget.
    pub(crate) fn prewarm(&self, sizes: &[usize], make: impl Fn(usize) -> T) {
        let limit = budget().buffers;
        let mut free = self.free.borrow_mut();
        for &bytes in sizes.iter().take(limit.saturating_sub(free.len())) {
            free.push(make(bytes));
        }
    }

    pub(crate) fn clear(&self) {
        self.free.borrow_mut().clear();
    }

    /// `(live, retained, retained bytes)` — `bytes_of` measures one idle item.
    pub(crate) fn stats(&self, bytes_of: impl Fn(&T) -> usize) -> (usize, usize, usize) {
        let free = self.free.borrow();
        (self.live.get(), free.len(), free.iter().map(bytes_of).sum())
    }
}
