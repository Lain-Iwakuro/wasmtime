//! A Cranelift-specific x64 assembler.
//!
//! All instructions known to this assembler are listed in the [`inst`] module.
//! The [`Inst`] enumeration contains a variant for each, allowing matching over
//! all these instructions. All of this is parameterized by a [`Registers`]
//! trait, allowing users of this assembler to plug in their own register types.
//!
//! ```
//! # use cranelift_assembler_x64::{Fixed, Imm8, inst, Inst, Registers};
//! // Tell the assembler the type of registers we're using; we can always
//! // encode a HW register as a `u8` (e.g., `eax = 0`).
//! pub struct Regs;
//! impl Registers for Regs {
//!     type ReadGpr = u8;
//!     type ReadWriteGpr = u8;
//!     type WriteGpr = u8;
//!     type ReadXmm = u8;
//!     type ReadWriteXmm = u8;
//!     type WriteXmm = u8;
//! }
//!
//! // Then, build one of the `AND` instructions; this one operates on an
//! // implicit `AL` register with an immediate. We can collect a sequence of
//! // instructions by converting to the `Inst` type.
//! let rax: u8 = 0;
//! let and = inst::andb_i::new(Fixed(rax), Imm8::new(0b10101010));
//! let seq: Vec<Inst<Regs>> = vec![and.into()];
//!
//! // Now we can encode this sequence into a code buffer.
//! let mut buffer = vec![];
//! for inst in seq {
//!     inst.encode(&mut buffer);
//! }
//! assert_eq!(buffer, vec![0x24, 0b10101010]);
//! ```
//!
//! With an [`Inst`], we can encode the instruction into a code buffer; see the
//! [example](Inst).

#![allow(
    non_camel_case_types,
    reason = "all of the generated struct names use snake case"
)]
#![no_std]
extern crate alloc;
#[cfg(any(test, feature = "fuzz"))]
extern crate std;

mod api;
mod custom;
mod evex;
mod features;
mod fixed;
pub mod gpr;
mod imm;
pub mod inst;
mod mem;
mod rex;
mod vex;
pub mod xmm;

#[cfg(any(test, feature = "fuzz"))]
pub mod fuzz;

/// An assembly instruction; contains all instructions known to the assembler.
///
/// This wraps all [`inst`] structures into a single enumeration for collecting
/// instructions.
#[doc(inline)]
// This re-exports, and documents, a module that is more convenient to use at
// the library top-level.
pub use inst::Inst;

pub use api::{
    AsReg, CodeSink, Constant, KnownOffset, Label, RegisterVisitor, Registers, TrapCode,
};
pub use features::{AvailableFeatures, Feature, Features};
pub use fixed::Fixed;
pub use gpr::{Gpr, NonRspGpr, Size};
pub use imm::{Extension, Imm8, Imm16, Imm32, Imm64, Simm8, Simm16, Simm32};
pub use mem::{
    Amode, AmodeOffset, AmodeOffsetPlusKnownOffset, DeferredTarget, GprMem, Scale, XmmMem,
};
pub use rex::RexPrefix;
pub use xmm::Xmm;

/// Architectural memory effects, including implicit stack accesses.
/// Read/write means one indivisible instruction: a guard belongs before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAccess {
    /// No memory access (including LEA and NOP).
    None,
    /// Memory read, including folded operands and return addresses.
    Read,
    /// Memory write, including pushes.
    Write,
    /// Both a read and a possible write, including RMW and memory push/pop.
    ReadWrite,
    /// A call may read and write arbitrary memory.
    Call,
    /// An explicit LFENCE.
    Fence,
}

/// An actual explicit memory operand, before register write effects are applied.
/// `bytes` describes the memory operand, including narrow and scalar SIMD forms.
#[derive(Clone, Copy, Debug)]
pub struct MemoryOperand<'a, R: AsReg> {
    /// The actual encoded effective address.
    pub address: &'a Amode<R>,
    /// Read, write, or possible read/write of this operand.
    pub access: MemoryAccess,
    /// Number of bytes touched, before any destination extension.
    pub bytes: u16,
}

/// Implicit stack accesses, in addition to an instruction's explicit operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StackAccess {
    /// No implicit stack access.
    None,
    /// Store below the old RSP after decrementing it.
    Push {
        /// Number of bytes stored and subtracted from RSP.
        bytes: u16,
    },
    /// Read at the old RSP before incrementing it.
    Pop {
        /// Number of bytes read and added to RSP.
        bytes: u16,
    },
    /// Read the eight-byte return address at RSP.
    Return,
    /// Store the eight-byte return address and enter an opaque callee.
    Call,
}
