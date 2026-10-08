//! Classification of AArch64 instructions for the SSBB store-to-load rule.
//!
//! Emission consults these same functions. An instruction whose expansion
//! contains a store followed by a load sets `needs_internal`, and the emitter
//! places the barrier between those operations.

use super::Inst;
use super::emit::return_call_setup_insts;
use crate::CodegenError;
use crate::CodegenResult;
use crate::ir::Type;
use crate::isa::aarch64::abi::AArch64MachineDeps;
use crate::machinst::abi::{ABIMachineSpec, CallInfo, RetLocation, StackAMode};
use crate::machinst::ssbb::{SsbbMemStep, SsbbSeq};
use crate::machinst::{FrameLayout, MachInst};
use crate::settings::Flags;
use alloc::string::String;
use smallvec::SmallVec;

/// `DSB #0`, the architectural encoding of SSBB.
/// Apple's `otool` disassembles this word as `ssbb`.
pub(crate) const SSBB_ENCODING: u32 = 0xD503309F;

pub(crate) enum RetvalOp {
    Island { needed_space: u32 },
    Mem { step: SsbbMemStep, inst: Inst },
}

/// Stack-carried return values, in the same order as `emit_retval_loads`.
pub(crate) fn call_retval_ops<T>(
    info: &CallInfo<T>,
    stackslots_size: u32,
) -> SmallVec<[RetvalOp; 4]> {
    let mut ops = SmallVec::new();
    let mut space_needed = 0u32;
    for pair in info.defs.iter() {
        if matches!(pair.location, RetLocation::Stack(..)) {
            // Same budget `emit_retval_loads` uses before the loads.
            space_needed += 10 * Inst::worst_case_size();
        }
    }
    if space_needed > 0 {
        ops.push(RetvalOp::Island {
            needed_space: space_needed,
        });
    }

    let temp = AArch64MachineDeps::retval_temp_reg(info.callee_conv);
    let word = AArch64MachineDeps::word_bytes();
    for pair in info.defs.iter() {
        match &pair.location {
            RetLocation::Reg(..) => {}
            RetLocation::Stack(amode, ty) => {
                if let Some(spillslot) = pair.vreg.to_reg().to_spillslot() {
                    let parts = (ty.bytes() + word - 1) / word;
                    let one_part_load_ty =
                        Type::int_with_byte_size(word.min(ty.bytes()) as u16).unwrap();
                    for part in 0..parts {
                        ops.push(RetvalOp::Mem {
                            step: SsbbMemStep::Load,
                            inst: AArch64MachineDeps::gen_load_stack(
                                amode.offset_by(part * word),
                                temp,
                                one_part_load_ty,
                            ),
                        });
                        ops.push(RetvalOp::Mem {
                            step: SsbbMemStep::Store,
                            inst: AArch64MachineDeps::gen_store_stack(
                                StackAMode::Slot(
                                    i64::from(stackslots_size)
                                        + i64::from(word)
                                            * ((spillslot.index() as i64) + (part as i64)),
                                ),
                                temp.to_reg(),
                                AArch64MachineDeps::word_type(),
                            ),
                        });
                    }
                } else {
                    ops.push(RetvalOp::Mem {
                        step: SsbbMemStep::Load,
                        inst: AArch64MachineDeps::gen_load_stack(*amode, pair.vreg, *ty),
                    });
                }
            }
        }
    }
    ops
}

pub(crate) fn classify(inst: &Inst, frame: &FrameLayout, flags: &Flags) -> CodegenResult<SsbbSeq> {
    match inst {
        Inst::ULoad8 { .. }
        | Inst::SLoad8 { .. }
        | Inst::ULoad16 { .. }
        | Inst::SLoad16 { .. }
        | Inst::ULoad32 { .. }
        | Inst::SLoad32 { .. }
        | Inst::ULoad64 { .. }
        | Inst::LoadP64 { .. }
        | Inst::FpuLoad16 { .. }
        | Inst::FpuLoad32 { .. }
        | Inst::FpuLoad64 { .. }
        | Inst::FpuLoad128 { .. }
        | Inst::FpuLoadP64 { .. }
        | Inst::FpuLoadP128 { .. }
        | Inst::LoadAcquire { .. }
        | Inst::VecLoadReplicate { .. }
        | Inst::LoadExtNameGot { .. }
        | Inst::LoadExtNameFar { .. }
        | Inst::JTSequence { .. } => one(SsbbMemStep::Load),

        Inst::Store8 { .. }
        | Inst::Store16 { .. }
        | Inst::Store32 { .. }
        | Inst::Store64 { .. }
        | Inst::StoreP64 { .. }
        | Inst::FpuStore16 { .. }
        | Inst::FpuStore32 { .. }
        | Inst::FpuStore64 { .. }
        | Inst::FpuStore128 { .. }
        | Inst::FpuStoreP64 { .. }
        | Inst::FpuStoreP128 { .. }
        | Inst::StoreRelease { .. }
        | Inst::StackProbeLoop { .. } => one(SsbbMemStep::Store),

        Inst::AtomicRMW { .. } | Inst::AtomicCAS { .. } => one(SsbbMemStep::Atomic),

        Inst::AtomicRMWLoop { .. } | Inst::AtomicCASLoop { .. } => {
            Err(CodegenError::Unsupported(String::from(
                "aarch64_mitigation=ssbb does not cover LL/SC atomic sequences; native LSE atomics require has_lse",
            )))
        }

        Inst::Ssbb { .. } => one(SsbbMemStep::Clear),

        // Existing CSDB and DMB do not separate a store from a later load.
        Inst::Csdb { .. } | Inst::Fence { .. } => Ok(SsbbSeq::neutral()),

        Inst::Call { info } => Ok(call_seq(info, frame.stackslots_size)),
        Inst::CallInd { info } => Ok(call_seq(info, frame.stackslots_size)),
        Inst::ReturnCall { info } => tail_seq(info, flags, frame),
        Inst::ReturnCallInd { info } => tail_seq(info, flags, frame),

        // adrp + ldr, then the helper call. The load runs first.
        Inst::ElfTlsGetAddr { .. } => loads_then_call(1),
        // adrp, a raw literal load, another load, then the helper call.
        Inst::MachOTlsGetAddr { .. } => loads_then_call(2),

        Inst::Nop0 { .. }
        | Inst::Nop4 { .. }
        | Inst::AluRRR { .. }
        | Inst::AluRRRR { .. }
        | Inst::AluRRImm12 { .. }
        | Inst::AluRRImmLogic { .. }
        | Inst::AluRRImmShift { .. }
        | Inst::AluRRRShift { .. }
        | Inst::AluRRRExtend { .. }
        | Inst::BitRR { .. }
        | Inst::Mov { .. }
        | Inst::MovFromPReg { .. }
        | Inst::MovToPReg { .. }
        | Inst::MovWide { .. }
        | Inst::MovK { .. }
        | Inst::Extend { .. }
        | Inst::CSel { .. }
        | Inst::CSNeg { .. }
        | Inst::CSet { .. }
        | Inst::CSetm { .. }
        | Inst::CCmp { .. }
        | Inst::CCmpImm { .. }
        | Inst::FpuMove32 { .. }
        | Inst::FpuMove64 { .. }
        | Inst::FpuMove128 { .. }
        | Inst::FpuMoveFromVec { .. }
        | Inst::FpuExtend { .. }
        | Inst::FpuRR { .. }
        | Inst::FpuRRR { .. }
        | Inst::FpuRRI { .. }
        | Inst::FpuRRIMod { .. }
        | Inst::FpuRRRR { .. }
        | Inst::FpuCmp { .. }
        | Inst::FpuToInt { .. }
        | Inst::IntToFpu { .. }
        | Inst::FpuCSel16 { .. }
        | Inst::FpuCSel32 { .. }
        | Inst::FpuCSel64 { .. }
        | Inst::FpuRound { .. }
        | Inst::MovToFpu { .. }
        | Inst::FpuMoveFPImm { .. }
        | Inst::MovToVec { .. }
        | Inst::MovFromVec { .. }
        | Inst::MovFromVecSigned { .. }
        | Inst::VecDup { .. }
        | Inst::VecDupFromFpu { .. }
        | Inst::VecDupFPImm { .. }
        | Inst::VecDupImm { .. }
        | Inst::VecExtend { .. }
        | Inst::VecMovElement { .. }
        | Inst::VecRRLong { .. }
        | Inst::VecRRNarrowLow { .. }
        | Inst::VecRRNarrowHigh { .. }
        | Inst::VecRRPair { .. }
        | Inst::VecRRRLong { .. }
        | Inst::VecRRRLongMod { .. }
        | Inst::VecRRPairLong { .. }
        | Inst::VecRRR { .. }
        | Inst::VecRRRMod { .. }
        | Inst::VecFmlaElem { .. }
        | Inst::VecMisc { .. }
        | Inst::VecLanes { .. }
        | Inst::VecShiftImm { .. }
        | Inst::VecShiftImmMod { .. }
        | Inst::VecExtract { .. }
        | Inst::VecTbl { .. }
        | Inst::VecTblExt { .. }
        | Inst::VecTbl2 { .. }
        | Inst::VecTbl2Ext { .. }
        | Inst::VecCSel { .. }
        | Inst::MovToNZCV { .. }
        | Inst::MovFromNZCV { .. }
        | Inst::Args { .. }
        | Inst::Rets { .. }
        | Inst::Ret { .. }
        | Inst::AuthenticatedRet { .. }
        | Inst::Jump { .. }
        | Inst::CondBr { .. }
        | Inst::TestBitAndBranch { .. }
        | Inst::TrapIf { .. }
        | Inst::IndirectBr { .. }
        | Inst::Brk { .. }
        | Inst::Udf { .. }
        | Inst::Adr { .. }
        | Inst::Adrp { .. }
        | Inst::Word4 { .. }
        | Inst::Word8 { .. }
        | Inst::LoadExtNameNear { .. }
        | Inst::LoadAddr { .. }
        | Inst::Paci { .. }
        | Inst::Xpaclri { .. }
        | Inst::Bti { .. }
        | Inst::EmitIsland { .. }
        | Inst::Unwind { .. }
        | Inst::DummyUse { .. }
        | Inst::LabelAddress { .. }
        | Inst::SequencePoint { .. } => Ok(SsbbSeq::neutral()),
    }
}

fn one(step: SsbbMemStep) -> CodegenResult<SsbbSeq> {
    let mut seq = SsbbSeq::neutral();
    seq.steps.push(step);
    Ok(seq)
}

fn loads_then_call(loads: usize) -> CodegenResult<SsbbSeq> {
    let mut seq = SsbbSeq::neutral();
    for _ in 0..loads {
        seq.steps.push(SsbbMemStep::Load);
    }
    seq.steps.push(SsbbMemStep::Call);
    Ok(seq)
}

fn call_seq<T>(info: &CallInfo<T>, stackslots_size: u32) -> SsbbSeq {
    let mut seq = SsbbSeq::neutral();
    seq.steps.push(SsbbMemStep::Call);
    if info.try_call_info.is_some() {
        seq.exn_cut = Some(0);
    }
    for op in call_retval_ops(info, stackslots_size) {
        if let RetvalOp::Mem { step, .. } = op {
            seq.steps.push(step);
        }
    }
    seq
}

fn tail_seq<T>(
    info: &super::ReturnCallInfo<T>,
    flags: &Flags,
    frame: &FrameLayout,
) -> CodegenResult<SsbbSeq> {
    let mut seq = SsbbSeq::neutral();
    for inst in return_call_setup_insts(frame, flags, info) {
        let part = classify(&inst, frame, flags)?;
        if part.needs_internal() {
            return Err(CodegenError::Unsupported(String::from(
                "aarch64_mitigation=ssbb: tail-call restore has an internal store-to-load",
            )));
        }
        seq.steps.extend(part.steps);
    }
    seq.steps.push(SsbbMemStep::Call);
    Ok(seq)
}

#[cfg(test)]
#[path = "ssbb_tests.rs"]
mod tests;
