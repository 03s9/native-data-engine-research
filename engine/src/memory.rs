//! SLICE D1. Hierarchical memory budget (spec §6.1). Thread-safe, lock-free accounting.
//!
//! Each level keeps its own `used` counter. A reservation is applied to the level and every
//! ancestor, all-or-nothing: a failure at any level rolls back the levels already charged.
use crate::types::{EngineError, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub struct MemoryBudget {
    used: AtomicUsize,
    limit: usize,
    parent: Option<Arc<MemoryBudget>>,
}

/// RAII reservation: bytes returned to the budget (and parent chain) on drop.
#[derive(Debug)]
pub struct Reservation {
    budget: Arc<MemoryBudget>,
    size: usize,
}

impl MemoryBudget {
    pub fn new(limit_bytes: usize) -> Arc<MemoryBudget> {
        Arc::new(MemoryBudget { used: AtomicUsize::new(0), limit: limit_bytes, parent: None })
    }

    /// Child budget: its reservations also count against the parent; `limit_bytes` caps the child.
    pub fn child(self: &Arc<Self>, limit_bytes: usize) -> Arc<MemoryBudget> {
        Arc::new(MemoryBudget {
            used: AtomicUsize::new(0),
            limit: limit_bytes,
            parent: Some(Arc::clone(self)),
        })
    }

    /// Err(EngineError::Memory{..}) when the reservation would exceed this or any ancestor limit. Never partial.
    pub fn try_reserve(self: &Arc<Self>, bytes: usize) -> Result<Reservation> {
        self.reserve_chain(bytes)?;
        Ok(Reservation { budget: Arc::clone(self), size: bytes })
    }

    /// Bytes charged at this level (includes reservations made through child budgets).
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// This level's own limit.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Smallest headroom along the chain from this level to the root.
    pub fn available(&self) -> usize {
        let own = self.limit.saturating_sub(self.used());
        match &self.parent {
            Some(parent) => own.min(parent.available()),
            None => own,
        }
    }

    /// Charge `bytes` at every level from here up to the root, or charge nothing.
    fn reserve_chain(&self, bytes: usize) -> Result<()> {
        self.reserve_level(bytes)?;
        if let Some(parent) = &self.parent {
            if let Err(err) = parent.reserve_chain(bytes) {
                self.release_level(bytes);
                return Err(err);
            }
        }
        Ok(())
    }

    /// Single-level CAS loop: the limit is checked against the exact value it swaps out.
    fn reserve_level(&self, bytes: usize) -> Result<()> {
        let mut current = self.used.load(Ordering::Relaxed);
        loop {
            let next = current.checked_add(bytes).filter(|&n| n <= self.limit);
            let Some(next) = next else {
                return Err(EngineError::Memory {
                    requested: bytes,
                    available: self.limit.saturating_sub(current),
                });
            };
            match self.used.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return Ok(()),
                Err(actual) => current = actual,
            }
        }
    }

    fn release_level(&self, bytes: usize) {
        let prev = self.used.fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(prev >= bytes, "memory budget released more than reserved");
    }

    fn release_chain(&self, bytes: usize) {
        self.release_level(bytes);
        if let Some(parent) = &self.parent {
            parent.release_chain(bytes);
        }
    }
}

impl Reservation {
    pub fn size(&self) -> usize {
        self.size
    }

    /// Grow in place; Err leaves the reservation unchanged.
    pub fn grow(&mut self, more: usize) -> Result<()> {
        self.budget.reserve_chain(more)?;
        // Charged successfully, so size + more <= used <= limit: no overflow possible.
        self.size += more;
        Ok(())
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.size > 0 {
            self.budget.release_chain(self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::sync::atomic::AtomicBool;

    #[test]
    fn reserve_and_release_restores_usage() {
        let b = MemoryBudget::new(100);
        let r = b.try_reserve(40).unwrap();
        assert_eq!((r.size(), b.used(), b.available()), (40, 40, 60));
        drop(r);
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn over_limit_reserve_fails_and_leaves_no_partial_state() {
        let b = MemoryBudget::new(100);
        let _held = b.try_reserve(70).unwrap();
        let err = b.try_reserve(31).unwrap_err();
        assert_eq!(err, EngineError::Memory { requested: 31, available: 30 });
        assert_eq!(b.used(), 70);
    }

    #[test]
    fn exact_limit_is_allowed_and_zero_byte_reserve_works() {
        let b = MemoryBudget::new(8);
        let zero = b.try_reserve(0).unwrap();
        let full = b.try_reserve(8).unwrap();
        assert_eq!(b.available(), 0);
        assert!(b.try_reserve(1).is_err());
        drop((zero, full));
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn grow_success_updates_size_and_usage() {
        let b = MemoryBudget::new(100);
        let mut r = b.try_reserve(10).unwrap();
        r.grow(25).unwrap();
        assert_eq!((r.size(), b.used()), (35, 35));
        drop(r);
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn grow_failure_leaves_reservation_unchanged() {
        let b = MemoryBudget::new(50);
        let mut r = b.try_reserve(30).unwrap();
        assert!(r.grow(21).is_err());
        assert_eq!((r.size(), b.used()), (30, 30));
        assert!(r.grow(usize::MAX).is_err());
        assert_eq!((r.size(), b.used()), (30, 30));
    }

    #[test]
    fn child_charges_parent_and_release_walks_up_the_chain() {
        let parent = MemoryBudget::new(100);
        let child = parent.child(60);
        let r = child.try_reserve(50).unwrap();
        assert_eq!((child.used(), parent.used()), (50, 50));
        assert_eq!(parent.available(), 50);
        assert_eq!(child.available(), 10);
        drop(r);
        assert_eq!((child.used(), parent.used()), (0, 0));
    }

    #[test]
    fn child_limit_caps_even_when_parent_has_room() {
        let parent = MemoryBudget::new(1000);
        let child = parent.child(10);
        assert!(child.try_reserve(11).is_err());
        assert_eq!((child.used(), parent.used()), (0, 0));
    }

    #[test]
    fn failed_child_reservation_rolls_back_parent_charge() {
        let parent = MemoryBudget::new(20);
        let child = parent.child(100);
        let _held = parent.try_reserve(15).unwrap();
        // Child has room, parent does not: the child's own charge must be rolled back too.
        assert!(child.try_reserve(10).is_err());
        assert_eq!((child.used(), parent.used()), (0, 15));
        assert!(child.try_reserve(5).is_ok());
    }

    #[test]
    fn grandchild_respects_every_ancestor() {
        let root = MemoryBudget::new(100);
        let mid = root.child(40);
        let leaf = mid.child(90);
        let r = leaf.try_reserve(40).unwrap();
        assert!(leaf.try_reserve(1).is_err());
        assert_eq!(root.used(), 40);
        drop(r);
        assert_eq!((mid.used(), root.used()), (0, 0));
    }

    #[test]
    fn reservation_outliving_its_child_handle_still_releases() {
        let parent = MemoryBudget::new(100);
        let r = {
            let child = parent.child(100);
            child.try_reserve(30).unwrap()
        };
        assert_eq!(parent.used(), 30);
        drop(r);
        assert_eq!(parent.used(), 0);
    }

    #[test]
    fn contention_eight_threads_never_exceed_limit() {
        const LIMIT: usize = 1000;
        let budget = MemoryBudget::new(LIMIT);
        let held_total = Arc::new(AtomicUsize::new(0));
        let violated = Arc::new(AtomicBool::new(false));
        let handles: Vec<_> = (0..8u64)
            .map(|seed| {
                let budget = Arc::clone(&budget);
                let held_total = Arc::clone(&held_total);
                let violated = Arc::clone(&violated);
                std::thread::spawn(move || {
                    let mut rng = StdRng::seed_from_u64(seed);
                    for _ in 0..5_000 {
                        let want = rng.gen_range(1..=200usize);
                        if let Ok(r) = budget.try_reserve(want) {
                            // Live reservations always sum to <= limit, and this thread's
                            // bytes are added only after its reservation succeeded.
                            let now = held_total.fetch_add(want, Ordering::SeqCst) + want;
                            if now > LIMIT || budget.used() > LIMIT {
                                violated.store(true, Ordering::SeqCst);
                            }
                            held_total.fetch_sub(want, Ordering::SeqCst);
                            drop(r);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(!violated.load(Ordering::SeqCst), "limit exceeded under contention");
        assert_eq!(budget.used(), 0);
        assert_eq!(held_total.load(Ordering::SeqCst), 0);
    }
}
