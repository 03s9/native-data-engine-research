//! SLICE C. Morsel-driven Rayon runtime + Tokio->Rayon bridge with credit-based backpressure (spec §3, §4).
use crate::column::Batch;
use crate::types::{EngineError, Result};

/// Split into morsels of at most `morsel_rows` rows (zero-row input -> empty vec).
pub fn split_morsels(_batch: &Batch, _morsel_rows: usize) -> Vec<Batch> { vec![] }

/// Run `op` over every morsel on a dedicated Rayon pool of `threads` workers; results returned in input order.
/// First error aborts remaining work and is returned. Panics in `op` become EngineError::Io("worker panic").
pub fn run_morsels<F>(_morsels: Vec<Batch>, _threads: usize, _op: F) -> Result<Vec<Batch>>
where F: Fn(&Batch) -> Result<Batch> + Send + Sync {
    Err(EngineError::Unimplemented("sched::run_morsels"))
}

/// Streaming dispatcher: async producers send batches via a bounded channel (capacity = credits);
/// Rayon workers never block waiting for I/O; `send` awaits when credits are exhausted (backpressure).
pub struct BridgeSender { _private: () }
pub struct BridgeReceiver { _private: () }
pub fn bridge(_credits: usize) -> (BridgeSender, BridgeReceiver) { unimplemented!("sched::bridge") }
impl BridgeSender {
    pub async fn send(&self, _b: Batch) -> Result<()> { Err(EngineError::Unimplemented("bridge send")) }
}
impl BridgeReceiver {
    /// Blocking pull for Rayon/sync side; None when all senders dropped and queue drained.
    pub fn recv(&self) -> Option<Batch> { None }
}
