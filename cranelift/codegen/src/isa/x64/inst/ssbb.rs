//! Experimental Spectre-v4 store-to-load classification after register allocation.
//!
//! Ordinary instructions use assembler operand metadata. This match deliberately
//! has no fallback: new pseudo instructions require a memory-effect audit.
//! Internal retry loops import a dirty backedge at their loop header. Their
//! Store marker is a compiler state update, not an additional machine store.

use super::*;
use crate::machinst::ssbb::{SsbbMemStep as Step, SsbbSeq};

pub(crate) fn classify(
    inst: &Inst,
    frame: &FrameLayout,
    info: &EmitInfo,
) -> CodegenResult<SsbbSeq> {
    let steps: SmallVec<[Step; 4]> = match inst {
        Inst::External { inst } => match inst.memory_access() {
            asm::MemoryAccess::None => smallvec![],
            asm::MemoryAccess::Read => smallvec![Step::Load],
            asm::MemoryAccess::Write => smallvec![Step::Store],
            asm::MemoryAccess::ReadWrite => smallvec![Step::Atomic],
            asm::MemoryAccess::Call => smallvec![Step::Call],
            asm::MemoryAccess::Fence => smallvec![Step::Clear],
        },
        Inst::CallKnown { info: call } => return call_seq(call, frame, info, !call.patchable),
        Inst::CallUnknown { info: call } => return call_seq(call, frame, info, true),
        Inst::ReturnCallKnown { info: call } => return tail_seq(call, frame, info),
        Inst::ReturnCallUnknown { info: call } => return tail_seq(call, frame, info),
        Inst::StackProbeLoop { .. } => smallvec![Step::Store],
        Inst::AtomicRmwSeq { .. } => smallvec![Step::Load, Step::Store, Step::Atomic],
        Inst::Atomic128RmwSeq { .. } | Inst::Atomic128XchgSeq { .. } => {
            smallvec![Step::Load, Step::Load, Step::Store, Step::Atomic]
        }
        Inst::LoadExtName { .. } => {
            if info.flags.is_pic() {
                smallvec![Step::Load]
            } else {
                smallvec![]
            }
        }
        Inst::JmpTableSeq { .. } | Inst::CoffTlsGetAddr { .. } => smallvec![Step::Load],
        // Guard before the whole ELF linker-recognized LEA/CALL template.
        Inst::ElfTlsGetAddr { .. } => smallvec![Step::Call],
        Inst::MachOTlsGetAddr { .. } => smallvec![Step::Load, Step::Call],
        Inst::StackSwitchBasic { .. } | Inst::WinchJmpIf { .. } => {
            return Err(CodegenError::Unsupported(format!(
                "x64_mitigation=lfence does not cover stack switching or non-CFG Winch branches: {inst:?}"
            )));
        }
        // These expansions contain only register operations/control flow. FP
        // conversion constants are materialized in GPRs, never loaded from a pool.
        Inst::CheckedSRemSeq { .. }
        | Inst::CheckedSRemSeq8 { .. }
        | Inst::MovFromPReg { .. }
        | Inst::MovToPReg { .. }
        | Inst::XmmCmove { .. }
        | Inst::XmmMinMaxSeq { .. }
        | Inst::CvtUint64ToFloatSeq { .. }
        | Inst::CvtFloatToSintSeq { .. }
        | Inst::CvtFloatToUintSeq { .. }
        | Inst::XmmUninitializedValue { .. }
        | Inst::GprUninitializedValue { .. }
        | Inst::Args { .. }
        | Inst::Rets { .. }
        | Inst::JmpKnown { .. }
        | Inst::JmpCond { .. }
        | Inst::JmpCondOr { .. }
        | Inst::TrapIf { .. }
        | Inst::TrapIfAnd { .. }
        | Inst::TrapIfOr { .. }
        | Inst::Unwind { .. }
        | Inst::DummyUse { .. }
        | Inst::LabelAddress { .. }
        | Inst::SequencePoint { .. } => smallvec![],
    };
    Ok(SsbbSeq {
        steps,
        exn_cut: None,
    })
}

fn call_seq<T>(
    call: &CallInfo<T>,
    frame: &FrameLayout,
    info: &EmitInfo,
    load_returns: bool,
) -> CodegenResult<SsbbSeq> {
    let mut seq = SsbbSeq {
        steps: smallvec![Step::Call],
        exn_cut: call.try_call_info.as_ref().map(|_| 0),
    };
    if load_returns {
        let mut insts = Vec::new();
        call.emit_retval_loads::<X64ABIMachineSpec, _, _>(
            frame.stackslots_size,
            |inst| insts.push(inst),
            |_| None,
        );
        for inst in insts {
            seq.steps.extend(classify(&inst, frame, info)?.steps);
        }
    }
    Ok(seq)
}

fn tail_seq<T>(
    call: &ReturnCallInfo<T>,
    frame: &FrameLayout,
    info: &EmitInfo,
) -> CodegenResult<SsbbSeq> {
    let mut seq = SsbbSeq::neutral();
    for inst in emit::return_call_setup_insts(call, frame, info) {
        seq.steps.extend(classify(&inst, frame, info)?.steps);
    }
    seq.steps.push(Step::Call);
    Ok(seq)
}
