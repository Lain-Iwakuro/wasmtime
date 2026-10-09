//! Experimental Spectre-v4 store-to-load classification after register allocation.
//!
//! Ordinary instructions use assembler operand metadata. This match deliberately
//! has no fallback: new pseudo instructions require a memory-effect audit.
//! Internal retry loops import a dirty backedge at their loop header. Their
//! Store marker is a compiler state update, not an additional machine store.

use super::external::AsmInst;
use super::*;
use crate::machinst::ssbb::{SsbbMemStep as Step, SsbbSeq};
use crate::machinst::v4_fences::{V4BarrierOrigin, V4Context, V4Event, V4Range, V4Seq};

/// Facts for the frozen final instruction stream. Effective addresses always
/// use old register versions; definitions are appended only afterward.
pub(crate) fn classify_v4(
    inst: &Inst,
    frame: &FrameLayout,
    info: &EmitInfo,
    context: &mut V4Context,
) -> CodegenResult<V4Seq> {
    if context.external_entry {
        context.set_frame_offset(regs::rsp(), Some(0));
        context.set_frame_offset(regs::rbp(), None);
        context.external_entry = false;
    }
    if let Inst::External { inst } = inst {
        return external_v4(inst, frame, context);
    }

    // The outer plan protects the first access. Freeze only guards required
    // inside an expansion, such as call -> stack return load or a retry edge.
    // Linker templates and complex expansions retain unknown address summaries.
    let mut seq = match inst {
        Inst::CallKnown { info: call } => call_v4(call, frame, info, !call.patchable)?,
        Inst::CallUnknown { info: call } => call_v4(call, frame, info, true)?,
        Inst::ReturnCallKnown { .. }
        | Inst::ReturnCallUnknown { .. }
        | Inst::AtomicRmwSeq { .. }
        | Inst::Atomic128RmwSeq { .. }
        | Inst::Atomic128XchgSeq { .. }
        | Inst::ElfTlsGetAddr { .. }
        | Inst::MachOTlsGetAddr { .. }
        | Inst::CoffTlsGetAddr { .. } => V4Seq::frozen_legacy(&classify(inst, frame, info)?),
        Inst::StackProbeLoop { .. } => V4Seq {
            events: smallvec![V4Event::Store(V4Range::Unknown)],
            ..V4Seq::neutral()
        },
        Inst::LoadExtName { .. } if info.flags.is_pic() => V4Seq {
            events: smallvec![V4Event::Load(V4Range::Unknown)],
            ..V4Seq::neutral()
        },
        Inst::JmpTableSeq { .. } => V4Seq {
            events: smallvec![V4Event::Load(V4Range::Unknown)],
            ..V4Seq::neutral()
        },
        Inst::StackSwitchBasic { .. } | Inst::WinchJmpIf { .. } => {
            return Err(CodegenError::Unsupported(format!(
                "selective v4 fences do not cover stack switching or non-CFG Winch branches: {inst:?}"
            )));
        }
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
        | Inst::SequencePoint { .. }
        | Inst::LoadExtName { .. } => V4Seq::neutral(),
        Inst::External { .. } => unreachable!(),
    };
    if let Inst::CallUnknown { info: call } = inst {
        if let RegMem::Mem { addr } = &call.dest {
            // The target is fetched by the CALL instruction before entering
            // its opaque callee; it is a real eight-byte native load.
            let address: asm::Amode<Gpr> = addr.clone().into();
            seq.events
                .insert(0, V4Event::Load(address_range(&address, frame, context, 8)));
            seq.exn_cut = seq.exn_cut.map(|cut| cut + 1);
        }
    }
    if matches!(
        inst,
        Inst::AtomicRmwSeq { .. } | Inst::Atomic128RmwSeq { .. } | Inst::Atomic128XchgSeq { .. }
    ) {
        // The old Store marker imports the retry backedge's dirty bit. It is
        // not a native store and must never become a spurious range fact.
        seq.events
            .retain(|event| !matches!(event, V4Event::Store(_)));
    }
    if matches!(inst, Inst::CoffTlsGetAddr { .. }) {
        // RIP-relative index, GS-relative TLS vector, indexed TLS entry.
        seq.events.extend([V4Event::Load(V4Range::Unknown); 2]);
    }
    if matches!(inst, Inst::MachOTlsGetAddr { .. }) {
        let call = seq
            .events
            .iter()
            .position(|event| *event == V4Event::OpaqueCall)
            .unwrap();
        seq.events.insert(call, V4Event::Load(V4Range::Unknown));
    }
    let old_rsp = context.frame_offset(regs::rsp());
    let frame_copy = match inst {
        Inst::MovToPReg { src, dst } => {
            let dst: Reg = (*dst).into();
            Some((dst, context.frame_offset(src.to_reg())))
        }
        _ => None,
    };
    for reg in pseudo_gpr_defs(inst) {
        append_def(&mut seq, context, reg);
    }
    if let Some((reg, offset)) = frame_copy {
        if is_frame_reg(reg) {
            context.set_frame_offset(reg, offset);
        }
    }
    if matches!(inst, Inst::StackProbeLoop { .. }) {
        append_def(&mut seq, context, regs::rsp());
        // The probe loop restores RSP by the same constant it subtracted.
        context.set_frame_offset(regs::rsp(), old_rsp);
    }
    Ok(seq)
}

fn external_v4(
    inst: &AsmInst,
    frame: &FrameLayout,
    context: &mut V4Context,
) -> CodegenResult<V4Seq> {
    let mut seq = V4Seq::neutral();
    let stack = inst.stack_access();
    let frame_update = external_frame_update(inst, frame, context);
    let memory = inst.explicit_memory();
    let explicit_range = memory
        .as_ref()
        .map(|memory| address_range(memory.address, frame, context, u32::from(memory.bytes)));
    let stack_range = match stack {
        asm::StackAccess::Push { bytes } => {
            Some(stack_range(context, -i64::from(bytes), u32::from(bytes)))
        }
        asm::StackAccess::Pop { bytes } => Some(stack_range(context, 0, u32::from(bytes))),
        asm::StackAccess::Return => Some(stack_range(context, 0, 8)),
        asm::StackAccess::None | asm::StackAccess::Call => None,
    };
    if matches!(
        stack,
        asm::StackAccess::Pop { .. } | asm::StackAccess::Return
    ) {
        seq.events.push(V4Event::Load(stack_range.unwrap()));
    }
    if let Some(memory) = memory {
        let mut range = explicit_range.unwrap();
        // POP computes an RSP-based destination after incrementing RSP. Use
        // Unknown for the memory form; the implicit stack read remains exact.
        if matches!(stack, asm::StackAccess::Pop { .. }) {
            range = V4Range::Unknown;
        }
        seq.events.push(match memory.access {
            asm::MemoryAccess::Read => V4Event::Load(range),
            asm::MemoryAccess::Write => V4Event::Store(range),
            asm::MemoryAccess::ReadWrite => V4Event::Rmw(range, range),
            _ => {
                return Err(CodegenError::Unsupported(format!(
                    "unaudited x64 explicit memory effect: {inst:?}"
                )));
            }
        });
    }
    match stack {
        asm::StackAccess::Push { .. } => seq.events.push(V4Event::Store(stack_range.unwrap())),
        asm::StackAccess::Call => seq.events.push(V4Event::OpaqueCall),
        _ => {}
    }
    if inst.memory_access() == asm::MemoryAccess::Fence {
        seq.events.push(V4Event::Barrier(V4BarrierOrigin::Upstream));
    }
    // Metadata cannot silently lose an architectural memory effect.
    if seq.events.is_empty() && inst.memory_access() != asm::MemoryAccess::None {
        return Err(CodegenError::Unsupported(format!(
            "x64 memory instruction missing v4 events: {inst:?}"
        )));
    }
    for encoding in inst.written_gprs() {
        append_def(
            &mut seq,
            context,
            Reg::from_real_reg(regs::gpr_preg(encoding)),
        );
    }
    if let Some((reg, offset)) = frame_update {
        context.set_frame_offset(reg, offset);
    }
    Ok(seq)
}

fn append_def(seq: &mut V4Seq, context: &mut V4Context, reg: Reg) {
    if reg.class() == RegClass::Int && !seq.events.contains(&V4Event::RegisterDef(reg)) {
        seq.events.push(V4Event::RegisterDef(reg));
        context.def(reg);
    }
}

fn is_frame_reg(reg: Reg) -> bool {
    reg == regs::rsp() || reg == regs::rbp()
}

fn stack_range(context: &V4Context, offset: i64, width: u32) -> V4Range {
    // Do not manufacture a LocalBase relation for a dynamic/unknown RSP.
    context.frame_range(regs::rsp(), offset, width)
}

fn displacement(offset: &asm::AmodeOffsetPlusKnownOffset, frame: &FrameLayout) -> Option<i64> {
    let known = match offset.offset {
        None => 0,
        Some(external::offsets::KEY_INCOMING_ARG) => {
            i64::from(frame.tail_args_size) + i64::from(frame.setup_area_size)
        }
        Some(external::offsets::KEY_SLOT_OFFSET) => i64::from(frame.outgoing_args_size),
        Some(_) => return None,
    };
    // Match the emitted signed-32-bit displacement, including overflow checks.
    let value = known.checked_add(i64::from(offset.simm32.value()))?;
    i32::try_from(value).ok().map(i64::from)
}

fn address_range(
    address: &asm::Amode<Gpr>,
    frame: &FrameLayout,
    context: &V4Context,
    width: u32,
) -> V4Range {
    match address {
        asm::Amode::ImmReg { base, simm32, .. } => {
            let Some(offset) = displacement(simm32, frame) else {
                return V4Range::Unknown;
            };
            let reg = base.to_reg();
            if is_frame_reg(reg) {
                context.frame_range(reg, offset, width)
            } else {
                context.local_base(reg, offset, width)
            }
        }
        asm::Amode::ImmRegRegShift { .. } | asm::Amode::RipRelative { .. } => V4Range::Unknown,
    }
}

fn external_frame_update(
    inst: &AsmInst,
    frame: &FrameLayout,
    context: &V4Context,
) -> Option<(Reg, Option<i64>)> {
    use asm::inst::Inst as I;
    let copy = match inst {
        I::movq_mr(asm::inst::movq_mr {
            rm64: asm::GprMem::Gpr(dst),
            r64,
        }) => Some((dst.to_reg().to_reg(), r64.as_ref().to_reg())),
        I::movq_rm(asm::inst::movq_rm {
            r64,
            rm64: asm::GprMem::Gpr(src),
        }) => Some((r64.as_ref().to_reg().to_reg(), src.to_reg())),
        _ => None,
    };
    if let Some((dst, src)) = copy {
        return is_frame_reg(dst).then(|| (dst, context.frame_offset(src)));
    }
    if let I::leaq_rm(asm::inst::leaq_rm {
        r64,
        m64: asm::Amode::ImmReg { base, simm32, .. },
    }) = inst
    {
        let dst = r64.as_ref().to_reg().to_reg();
        return is_frame_reg(dst).then(|| {
            (
                dst,
                context
                    .frame_offset(base.to_reg())
                    .and_then(|value| value.checked_add(displacement(simm32, frame)?)),
            )
        });
    }
    let arithmetic = match inst {
        I::addq_mi_sxl(asm::inst::addq_mi_sxl {
            rm64: asm::GprMem::Gpr(reg),
            imm32,
        }) => Some((reg.write.to_reg().to_reg(), i64::from(imm32.value()))),
        I::subq_mi_sxl(asm::inst::subq_mi_sxl {
            rm64: asm::GprMem::Gpr(reg),
            imm32,
        }) => Some((reg.write.to_reg().to_reg(), -i64::from(imm32.value()))),
        I::addq_mi_sxb(asm::inst::addq_mi_sxb {
            rm64: asm::GprMem::Gpr(reg),
            imm8,
        }) => Some((reg.write.to_reg().to_reg(), i64::from(imm8.value()))),
        I::subq_mi_sxb(asm::inst::subq_mi_sxb {
            rm64: asm::GprMem::Gpr(reg),
            imm8,
        }) => Some((reg.write.to_reg().to_reg(), -i64::from(imm8.value()))),
        _ => None,
    };
    if let Some((reg, delta)) = arithmetic {
        return is_frame_reg(reg).then(|| {
            (
                reg,
                context
                    .frame_offset(reg)
                    .and_then(|value| value.checked_add(delta)),
            )
        });
    }
    let delta = match inst.stack_access() {
        asm::StackAccess::Push { bytes } => -i64::from(bytes),
        asm::StackAccess::Pop { bytes } => {
            if inst
                .written_gprs()
                .iter()
                .filter(|encoding| **encoding == asm::gpr::enc::RSP)
                .count()
                > 1
            {
                return None;
            }
            i64::from(bytes)
        }
        asm::StackAccess::Call => return Some((regs::rsp(), context.frame_offset(regs::rsp()))),
        _ => return None,
    };
    Some((
        regs::rsp(),
        context
            .frame_offset(regs::rsp())
            .and_then(|value| value.checked_add(delta)),
    ))
}

fn pseudo_gpr_defs(inst: &Inst) -> SmallVec<[Reg; 16]> {
    let mut defs = SmallVec::new();
    match inst {
        Inst::CheckedSRemSeq {
            dst_quotient,
            dst_remainder,
            ..
        } => defs.extend([
            dst_quotient.to_reg().to_reg(),
            dst_remainder.to_reg().to_reg(),
        ]),
        Inst::CheckedSRemSeq8 { dst, .. }
        | Inst::MovFromPReg { dst, .. }
        | Inst::GprUninitializedValue { dst }
        | Inst::LoadExtName { dst, .. }
        | Inst::LabelAddress { dst, .. } => defs.push(dst.to_reg().to_reg()),
        Inst::MovToPReg { dst, .. } => defs.push((*dst).into()),
        Inst::CvtUint64ToFloatSeq {
            tmp_gpr1, tmp_gpr2, ..
        } => defs.extend([tmp_gpr1.to_reg().to_reg(), tmp_gpr2.to_reg().to_reg()]),
        Inst::CvtFloatToSintSeq { dst, tmp_gpr, .. }
        | Inst::CvtFloatToUintSeq { dst, tmp_gpr, .. } => {
            defs.extend([dst.to_reg().to_reg(), tmp_gpr.to_reg().to_reg()])
        }
        Inst::StackProbeLoop { tmp, .. } => defs.push(tmp.to_reg()),
        Inst::JmpTableSeq { tmp1, tmp2, .. } => defs.extend([tmp1.to_reg(), tmp2.to_reg()]),
        Inst::AtomicRmwSeq { temp, dst_old, .. } => {
            defs.extend([temp.to_reg().to_reg(), dst_old.to_reg().to_reg()])
        }
        Inst::Atomic128RmwSeq {
            temp_low,
            temp_high,
            dst_old_low,
            dst_old_high,
            ..
        } => defs.extend([
            temp_low.to_reg().to_reg(),
            temp_high.to_reg().to_reg(),
            dst_old_low.to_reg().to_reg(),
            dst_old_high.to_reg().to_reg(),
        ]),
        Inst::Atomic128XchgSeq {
            dst_old_low,
            dst_old_high,
            ..
        } => defs.extend([
            dst_old_low.to_reg().to_reg(),
            dst_old_high.to_reg().to_reg(),
        ]),
        Inst::CallKnown { info } => call_defs(info, &mut defs),
        Inst::CallUnknown { info } => call_defs(info, &mut defs),
        Inst::ReturnCallKnown { .. } | Inst::ReturnCallUnknown { .. } => {
            // Callee-save restores and argument resizing redefine stack/FP
            // and restored GPRs. No frame facts survive this terminal expansion.
            defs.extend((0..16).map(|enc| Reg::from_real_reg(regs::gpr_preg(enc))));
        }
        Inst::Args { args } => defs.extend(args.iter().map(|arg| arg.preg)),
        Inst::ElfTlsGetAddr { .. } | Inst::MachOTlsGetAddr { .. } => {
            defs.extend(
                X64ABIMachineSpec::get_regs_clobbered_by_call(CallConv::SystemV, false)
                    .into_iter()
                    .map(Reg::from_real_reg),
            );
        }
        Inst::CoffTlsGetAddr { .. } => defs.extend([regs::rax(), regs::rcx()]),
        Inst::XmmCmove { .. }
        | Inst::XmmMinMaxSeq { .. }
        | Inst::XmmUninitializedValue { .. }
        | Inst::Rets { .. }
        | Inst::JmpKnown { .. }
        | Inst::JmpCond { .. }
        | Inst::JmpCondOr { .. }
        | Inst::TrapIf { .. }
        | Inst::TrapIfAnd { .. }
        | Inst::TrapIfOr { .. }
        | Inst::Unwind { .. }
        | Inst::DummyUse { .. }
        | Inst::SequencePoint { .. } => {}
        Inst::External { .. } | Inst::StackSwitchBasic { .. } | Inst::WinchJmpIf { .. } => {
            unreachable!()
        }
    }
    defs
}

fn call_defs<T>(info: &CallInfo<T>, defs: &mut SmallVec<[Reg; 16]>) {
    defs.extend(info.clobbers.into_iter().map(Reg::from_real_reg));
    for result in &info.defs {
        match result.location {
            RetLocation::Reg(reg, ..) => defs.push(reg),
            RetLocation::Stack(..) => defs.push(result.vreg.to_reg()),
        }
    }
}

fn call_v4<T>(
    call: &CallInfo<T>,
    frame: &FrameLayout,
    info: &EmitInfo,
    load_returns: bool,
) -> CodegenResult<V4Seq> {
    let legacy = call_seq(call, frame, info, load_returns)?;
    if legacy.needs_internal() {
        Ok(V4Seq::frozen_legacy(&legacy))
    } else {
        Ok(V4Seq {
            events: smallvec![V4Event::OpaqueCall],
            exn_cut: call.try_call_info.as_ref().map(|_| 0),
            ..V4Seq::neutral()
        })
    }
}

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
