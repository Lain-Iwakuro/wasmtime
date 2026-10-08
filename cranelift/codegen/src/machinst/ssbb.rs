#![allow(missing_docs)]

//! Store-to-load SSBB dataflow shared by instruction classification and emission.
//!
//! `dirty` means a store may have executed that is not yet separated by SSBB.
//! The bit exists only in the compiler.

use alloc::vec::Vec;
use smallvec::SmallVec;

/// One memory effect in emission order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsbbMemStep {
    /// Ordinary load, including acquire, narrow, paired, FP, and SIMD loads.
    Load,
    /// Ordinary store, including release and spill stores.
    Store,
    /// Atomic RMW or CAS: read, then maybe write.
    Atomic,
    /// Call that may read and may write.
    Call,
    /// An SSBB that is already in the instruction stream.
    Clear,
}

/// Ordered memory effects of one instruction, plus an optional exception cut.
#[derive(Clone, Debug)]
pub struct SsbbSeq {
    pub steps: SmallVec<[SsbbMemStep; 4]>,
    /// When set, exception successors observe state after this step (inclusive).
    /// The last successor observes state after every step.
    pub exn_cut: Option<u16>,
}

impl SsbbSeq {
    pub fn neutral() -> Self {
        Self {
            steps: SmallVec::new(),
            exn_cut: None,
        }
    }

    pub fn transfer(mut dirty: bool, steps: &[SsbbMemStep]) -> bool {
        for step in steps {
            dirty = Self::after(step);
        }
        dirty
    }

    fn after(step: &SsbbMemStep) -> bool {
        match step {
            SsbbMemStep::Load | SsbbMemStep::Clear => false,
            SsbbMemStep::Store | SsbbMemStep::Atomic | SsbbMemStep::Call => true,
        }
    }

    /// A barrier is required before this step when `dirty` is set.
    pub fn needs_guard(dirty: bool, step: SsbbMemStep) -> bool {
        dirty
            && matches!(
                step,
                SsbbMemStep::Load | SsbbMemStep::Atomic | SsbbMemStep::Call
            )
    }

    pub fn count_barriers(mut dirty: bool, steps: &[SsbbMemStep]) -> u32 {
        let mut count = 0;
        for step in steps {
            if Self::needs_guard(dirty, *step) {
                count += 1;
            }
            dirty = Self::after(step);
        }
        count
    }

    /// True when a barrier is required after the instruction has already started.
    /// A single SSBB before the whole instruction cannot implement that.
    pub fn needs_internal(&self) -> bool {
        let mut produced_dirty = false;
        for (i, step) in self.steps.iter().enumerate() {
            match step {
                SsbbMemStep::Store | SsbbMemStep::Atomic | SsbbMemStep::Call => {
                    if i > 0
                        && produced_dirty
                        && matches!(step, SsbbMemStep::Atomic | SsbbMemStep::Call)
                    {
                        return true;
                    }
                    produced_dirty = true;
                }
                SsbbMemStep::Load | SsbbMemStep::Clear => {
                    if produced_dirty {
                        return true;
                    }
                    produced_dirty = false;
                }
            }
        }
        false
    }
}

/// Per-block entry state and the exact number of SSBB bytes emission will insert.
pub struct SsbbPlan {
    pub block_in: Vec<bool>,
    pub extra_bytes: Vec<u32>,
}
