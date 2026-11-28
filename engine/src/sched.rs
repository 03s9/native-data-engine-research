//! SLICE C. Morsel-driven Rayon runtime + Tokio->Rayon bridge with credit-based backpressure (spec §3, §4).
//!
//! Design notes:
//! - The bridge is a single `tokio::sync::mpsc` channel of capacity `credits`. No forwarder thread is
//!   needed, so `bridge()` works without a runtime. `send` is async and parks only the task while credits
//!   are exhausted. `recv` is a blocking pull for the Rayon/sync side and must not be called from inside
//!   a Tokio runtime context (`blocking_recv` panics there).
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;

use crate::column::Batch;
use crate::types::{EngineError, Result};

/// Split into morsels of at most `morsel_rows` rows (zero-row input -> empty vec).
/// `morsel_rows == 0` is clamped to 1 so the split always terminates.
pub fn split_morsels(batch: &Batch, morsel_rows: usize) -> Vec<Batch> {
    let step = morsel_rows.max(1);
    (0..batch.num_rows)
        .step_by(step)
        .map(|offset| batch.slice(offset, step.min(batch.num_rows - offset)))
        .collect()
}

/// Run `op` over every morsel on a dedicated Rayon pool of `threads` workers; results returned in input order.
/// The first error (in time) aborts remaining work: morsels not yet started are skipped and that error is returned.
/// Panics in `op` become `EngineError::Io("worker panic")`. `threads == 0` is clamped to 1.
pub fn run_morsels<F>(morsels: Vec<Batch>, threads: usize, op: F) -> Result<Vec<Batch>>
where
    F: Fn(&Batch) -> Result<Batch> + Send + Sync,
{
    if morsels.is_empty() {
        return Ok(vec![]);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| EngineError::Io(format!("thread pool: {e}")))?;

    let abort = AtomicBool::new(false);
    let first_err: Mutex<Option<EngineError>> = Mutex::new(None);
    let record = |e: EngineError| {
        let mut slot = first_err.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some(e);
        }
        abort.store(true, Ordering::Release);
    };

    let outputs: Vec<Option<Batch>> = pool.install(|| {
        morsels
            .par_iter()
            .map(|m| {
                if abort.load(Ordering::Acquire) {
                    return None;
                }
                match catch_unwind(AssertUnwindSafe(|| op(m))) {
                    Ok(Ok(out)) => Some(out),
                    Ok(Err(e)) => {
                        record(e);
                        None
                    }
                    Err(_) => {
                        record(EngineError::Io("worker panic".into()));
                        None
                    }
                }
            })
            .collect()
    });

    if let Some(e) = first_err.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(e);
    }
    outputs
        .into_iter()
        .collect::<Option<Vec<Batch>>>()
        .ok_or_else(|| EngineError::Io("morsel result missing without error".into()))
}

/// Streaming dispatcher: async producers send batches via a bounded channel (capacity = credits);
/// Rayon workers never block waiting for I/O; `send` awaits when credits are exhausted (backpressure).
#[derive(Clone)]
pub struct BridgeSender {
    tx: tokio::sync::mpsc::Sender<Batch>,
}

pub struct BridgeReceiver {
    rx: Mutex<tokio::sync::mpsc::Receiver<Batch>>,
}

/// Creates a bridge with `credits` in-flight batches (clamped to at least 1).
pub fn bridge(credits: usize) -> (BridgeSender, BridgeReceiver) {
    let (tx, rx) = tokio::sync::mpsc::channel(credits.max(1));
    (BridgeSender { tx }, BridgeReceiver { rx: Mutex::new(rx) })
}

impl BridgeSender {
    /// Awaits a free credit. Errors if the receiver was dropped.
    pub async fn send(&self, b: Batch) -> Result<()> {
        self.tx
            .send(b)
            .await
            .map_err(|_| EngineError::Io("bridge receiver dropped".into()))
    }
}

impl BridgeReceiver {
    /// Blocking pull for Rayon/sync side; None when all senders dropped and queue drained.
    /// Must not be called from within a Tokio runtime context.
    pub fn recv(&self) -> Option<Batch> {
        let mut rx = self.rx.lock().unwrap_or_else(|p| p.into_inner());
        rx.blocking_recv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{schema_ref, Column};
    use crate::types::{DataType, Field, ScalarValue, Schema};
    use std::sync::atomic::{AtomicUsize, Ordering as AO};
    use std::sync::mpsc as std_mpsc;
    use std::sync::Arc;
    use std::time::Duration;

    const WATCHDOG: Duration = Duration::from_secs(20);

    fn int_batch(values: &[i64]) -> Batch {
        let schema = schema_ref(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        Batch::try_new(schema, vec![Column::from_i64(values.to_vec())]).unwrap()
    }

    fn ints(b: &Batch) -> Vec<i64> {
        (0..b.num_rows)
            .map(|i| match b.columns[0].get(i) {
                ScalarValue::Int64(v) => v,
                other => panic!("unexpected {other:?}"),
            })
            .collect()
    }

    /// Runs `f` on a helper thread and fails the test instead of hanging if it does not finish.
    fn with_watchdog<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std_mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(WATCHDOG)
            .expect("watchdog: operation did not finish (deadlock?)")
    }

    fn current_thread_rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn split_sizes_and_order() {
        let b = int_batch(&(0..10).collect::<Vec<i64>>());
        let parts = split_morsels(&b, 4);
        let sizes: Vec<usize> = parts.iter().map(|p| p.num_rows).collect();
        assert_eq!(sizes, vec![4, 4, 2]);
        let flat: Vec<i64> = parts.iter().flat_map(ints).collect();
        assert_eq!(flat, (0..10).collect::<Vec<i64>>());
    }

    #[test]
    fn split_edge_cases() {
        let empty = int_batch(&[]);
        assert!(split_morsels(&empty, 4).is_empty());
        let b = int_batch(&[1, 2, 3]);
        assert_eq!(split_morsels(&b, 3).len(), 1);
        assert_eq!(split_morsels(&b, 100)[0].num_rows, 3);
        // zero morsel size is clamped to 1 instead of looping forever
        assert_eq!(split_morsels(&b, 0).len(), 3);
    }

    #[test]
    fn run_morsels_preserves_order_under_uneven_sleeps() {
        let morsels: Vec<Batch> = (0..40).map(|i| int_batch(&[i])).collect();
        let out = run_morsels(morsels, 4, |m| {
            let v = ints(m)[0];
            // later morsels finish first when the early ones sleep longest
            std::thread::sleep(Duration::from_millis(((40 - v) % 7) as u64));
            Ok(m.clone())
        })
        .unwrap();
        let got: Vec<i64> = out.iter().flat_map(ints).collect();
        assert_eq!(got, (0..40).collect::<Vec<i64>>());
    }

    #[test]
    fn run_morsels_empty_input() {
        let out = run_morsels(vec![], 2, |m| Ok(m.clone())).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn run_morsels_first_error_aborts_remaining_work() {
        let morsels: Vec<Batch> = (0..200).map(|i| int_batch(&[i])).collect();
        let executed = AtomicUsize::new(0);
        let err = run_morsels(morsels, 2, |m| {
            executed.fetch_add(1, AO::SeqCst);
            let v = ints(m)[0];
            if v == 3 {
                return Err(EngineError::Arithmetic("boom at 3".into()));
            }
            std::thread::sleep(Duration::from_millis(1));
            Ok(m.clone())
        })
        .unwrap_err();
        assert_eq!(err, EngineError::Arithmetic("boom at 3".into()));
        let n = executed.load(AO::SeqCst);
        assert!(n < 200, "expected abort to skip work, executed {n} of 200");
    }

    #[test]
    fn run_morsels_panic_is_captured_as_io_error() {
        let morsels: Vec<Batch> = (0..8).map(|i| int_batch(&[i])).collect();
        let err = run_morsels(morsels, 3, |m| {
            if ints(m)[0] == 5 {
                panic!("intentional test panic");
            }
            Ok(m.clone())
        })
        .unwrap_err();
        assert_eq!(err, EngineError::Io("worker panic".into()));
    }

    #[test]
    fn bridge_backpressure_bounds_producer_run_ahead() {
        let credits = 2;
        let (tx, rx) = bridge(credits);
        let sent = Arc::new(AtomicUsize::new(0));
        let sent_p = sent.clone();
        let producer = std::thread::spawn(move || {
            current_thread_rt().block_on(async move {
                for i in 0..10 {
                    tx.send(int_batch(&[i])).await.unwrap();
                    sent_p.fetch_add(1, AO::SeqCst);
                }
            });
        });
        // consumer stalled: producer must stop at credits (queued) and cannot exceed credits + 1
        std::thread::sleep(Duration::from_millis(300));
        let stalled = sent.load(AO::SeqCst);
        assert!(
            stalled >= credits,
            "producer should have filled the credits, got {stalled}"
        );
        assert!(
            stalled <= credits + 1,
            "producer ran ahead: {stalled} > {}",
            credits + 1
        );

        let drained = with_watchdog(move || {
            let mut got = Vec::new();
            while let Some(b) = rx.recv() {
                got.extend(ints(&b));
            }
            got
        });
        producer.join().unwrap();
        assert_eq!(drained, (0..10).collect::<Vec<i64>>());
    }

    #[test]
    fn bridge_send_awaits_instead_of_blocking_the_runtime() {
        let (tx, _rx) = bridge(1);
        let rt = current_thread_rt();
        rt.block_on(async {
            tx.send(int_batch(&[1])).await.unwrap();
            // bridge is full: send must yield to the executor, not block the thread
            let r = tokio::time::timeout(Duration::from_millis(50), tx.send(int_batch(&[2]))).await;
            assert!(
                r.is_err(),
                "second send should be pending while no credit is free"
            );
        });
    }

    #[test]
    fn bridge_send_errors_after_receiver_dropped() {
        let (tx, rx) = bridge(2);
        drop(rx);
        let err = current_thread_rt()
            .block_on(tx.send(int_batch(&[1])))
            .unwrap_err();
        assert!(matches!(err, EngineError::Io(_)));
    }

    #[test]
    fn bridge_no_deadlock_when_consumer_is_rayon_worker() {
        let result = with_watchdog(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let (tx, rx) = bridge(2);
            let producer = std::thread::spawn(move || {
                current_thread_rt().block_on(async move {
                    for i in 0..50 {
                        tx.send(int_batch(&[i])).await.unwrap();
                    }
                });
            });
            let total = pool.install(|| {
                let mut count = 0usize;
                while let Some(b) = rx.recv() {
                    count += b.num_rows;
                }
                count
            });
            producer.join().unwrap();
            total
        });
        assert_eq!(result, 50);
    }

    #[test]
    fn bridge_drop_senders_terminates_receiver() {
        let (tx, rx) = bridge(4);
        let tx2 = tx.clone();
        let rt = current_thread_rt();
        rt.block_on(async {
            tx.send(int_batch(&[1])).await.unwrap();
            tx2.send(int_batch(&[2])).await.unwrap();
        });
        drop(tx);
        drop(tx2);
        let drained = with_watchdog(move || {
            let mut got = Vec::new();
            while let Some(b) = rx.recv() {
                got.extend(ints(&b));
            }
            got
        });
        assert_eq!(drained, vec![1, 2]);
    }

    #[test]
    fn bridge_thousand_batches_arrive_in_order() {
        let (tx, rx) = bridge(8);
        let producer = std::thread::spawn(move || {
            current_thread_rt().block_on(async move {
                for i in 0..1000 {
                    tx.send(int_batch(&[i, i])).await.unwrap();
                }
            });
        });
        let got = with_watchdog(move || {
            let mut out = Vec::new();
            while let Some(b) = rx.recv() {
                out.push(ints(&b)[0]);
            }
            out
        });
        producer.join().unwrap();
        assert_eq!(got, (0..1000).collect::<Vec<i64>>());
    }
}
