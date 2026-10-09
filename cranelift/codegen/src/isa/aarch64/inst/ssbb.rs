//! Classification of AArch64 instructions for the SSBB store-to-load rule.
//!
//! Emission consults these same functions. An instruction whose expansion
//! contains a store followed by a load sets `needs_internal`, and the emitter
//! places the barrier between those operations.

use super::Inst;
use super::emit::return_call_setup_insts;
use super::{
    ALUOp, AMode, OperandSize, PairAMode, Reg, fp_reg, link_reg, spilltmp_reg, stack_reg, tmp2_reg,
};
use crate::CodegenError;
use crate::CodegenResult;
use crate::ir::Type;
use crate::isa::aarch64::abi::AArch64MachineDeps;
use crate::machinst::abi::{ABIMachineSpec, CallInfo, RetLocation, StackAMode};
use crate::machinst::ssbb::{SsbbMemStep, SsbbSeq};
use crate::machinst::v4_fences::{V4BarrierOrigin, V4Context, V4Event, V4Range, V4Seq};
use crate::machinst::{FrameLayout, MachInst};
use crate::settings::Flags;
use alloc::string::String;
use smallvec::SmallVec;

/// Describe the same native accesses as `classify`, adding widths and the old
/// effective-address register version. Complex expansions keep their audited
/// conservative guards and may never be treated as a reordered load/store pair.
pub(crate) fn classify_v4(
    inst: &Inst,
    frame: &FrameLayout,
    flags: &Flags,
    context: &mut V4Context,
) -> CodegenResult<V4Seq> {
    let legacy = classify(inst, frame, flags)?;
    if context.external_entry {
        context.set_frame_offset(stack_reg(), Some(0));
        context.external_entry = false;
    }
    let mut seq = V4Seq::neutral();
    let mut sp_after = None;
    let mut frame_copy = None;
    match inst {
        Inst::ULoad8 { mem, .. } | Inst::SLoad8 { mem, .. } => {
            load(&mut seq, mem, 1, frame, context, &mut sp_after)
        }
        Inst::ULoad16 { mem, .. } | Inst::SLoad16 { mem, .. } | Inst::FpuLoad16 { mem, .. } => {
            load(&mut seq, mem, 2, frame, context, &mut sp_after)
        }
        Inst::ULoad32 { mem, .. } | Inst::SLoad32 { mem, .. } | Inst::FpuLoad32 { mem, .. } => {
            load(&mut seq, mem, 4, frame, context, &mut sp_after)
        }
        Inst::ULoad64 { mem, .. } | Inst::FpuLoad64 { mem, .. } => {
            load(&mut seq, mem, 8, frame, context, &mut sp_after)
        }
        Inst::FpuLoad128 { mem, .. } => load(&mut seq, mem, 16, frame, context, &mut sp_after),
        Inst::Store8 { mem, .. } => store(&mut seq, mem, 1, frame, context, &mut sp_after),
        Inst::Store16 { mem, .. } | Inst::FpuStore16 { mem, .. } => {
            store(&mut seq, mem, 2, frame, context, &mut sp_after)
        }
        Inst::Store32 { mem, .. } | Inst::FpuStore32 { mem, .. } => {
            store(&mut seq, mem, 4, frame, context, &mut sp_after)
        }
        Inst::Store64 { mem, .. } | Inst::FpuStore64 { mem, .. } => {
            store(&mut seq, mem, 8, frame, context, &mut sp_after)
        }
        Inst::FpuStore128 { mem, .. } => store(&mut seq, mem, 16, frame, context, &mut sp_after),
        Inst::LoadP64 { mem, .. } | Inst::FpuLoadP64 { mem, .. } => seq
            .events
            .push(V4Event::Load(pair_range(mem, 16, context, &mut sp_after))),
        Inst::FpuLoadP128 { mem, .. } => {
            seq.events
                .push(V4Event::Load(pair_range(mem, 32, context, &mut sp_after)))
        }
        Inst::StoreP64 { mem, .. } | Inst::FpuStoreP64 { mem, .. } => seq
            .events
            .push(V4Event::Store(pair_range(mem, 16, context, &mut sp_after))),
        Inst::FpuStoreP128 { mem, .. } => {
            seq.events
                .push(V4Event::Store(pair_range(mem, 32, context, &mut sp_after)))
        }
        Inst::LoadAcquire { rn, access_ty, .. } => seq.events.push(V4Event::Load(base_range(
            *rn,
            0,
            access_ty.bytes(),
            context,
        ))),
        Inst::StoreRelease { rn, access_ty, .. } => seq.events.push(V4Event::Store(base_range(
            *rn,
            0,
            access_ty.bytes(),
            context,
        ))),
        Inst::AtomicCAS { rn, ty, .. } | Inst::AtomicRMW { rn, ty, .. } => {
            let range = base_range(*rn, 0, ty.bytes(), context);
            seq.events.push(V4Event::Rmw(range, range));
        }
        Inst::VecLoadReplicate { rn, size, .. } => seq.events.push(V4Event::Load(base_range(
            *rn,
            0,
            size.lane_size().ty().bytes(),
            context,
        ))),
        Inst::StackProbeLoop { .. } => seq.events.push(V4Event::Store(V4Range::Unknown)),
        Inst::Ssbb { .. } => seq.events.push(V4Event::Barrier(V4BarrierOrigin::Upstream)),
        // Keep address-materialization/literal/table/TLS/tail templates together.
        // The shared plan protects their first read; only guards for transitions
        // within the expansion are fixed as legacy-internal barriers.
        Inst::JTSequence { .. }
        | Inst::LoadExtNameGot { .. }
        | Inst::LoadExtNameFar { .. }
        | Inst::ElfTlsGetAddr { .. }
        | Inst::MachOTlsGetAddr { .. }
        | Inst::ReturnCall { .. }
        | Inst::ReturnCallInd { .. } => seq = V4Seq::frozen_legacy(&legacy),
        Inst::Call { info } => {
            if info.callee_pop_size > 0 {
                sp_after = Some(context.frame_offset(stack_reg()));
            }
            if !info.patchable && legacy.needs_internal() {
                seq = V4Seq::frozen_legacy(&legacy);
            } else {
                seq.events.push(V4Event::OpaqueCall);
                if info.try_call_info.is_some() {
                    seq.exn_cut = Some(0);
                }
            }
        }
        Inst::CallInd { info } => {
            if info.callee_pop_size > 0 {
                sp_after = Some(context.frame_offset(stack_reg()));
            }
            if legacy.needs_internal() {
                seq = V4Seq::frozen_legacy(&legacy);
            } else {
                seq.events.push(V4Event::OpaqueCall);
                if info.try_call_info.is_some() {
                    seq.exn_cut = Some(0);
                }
            }
        }
        Inst::AluRRImm12 {
            alu_op,
            size: OperandSize::Size64,
            rd,
            rn,
            imm12,
        } if matches!(alu_op, ALUOp::Add | ALUOp::Sub)
            && (rd.to_reg() == stack_reg() || rd.to_reg() == fp_reg()) =>
        {
            let delta = i64::from(imm12.value()) * if *alu_op == ALUOp::Sub { -1 } else { 1 };
            frame_copy = Some((
                rd.to_reg(),
                context
                    .frame_offset(*rn)
                    .and_then(|off| off.checked_add(delta)),
            ));
        }
        Inst::Mov {
            size: OperandSize::Size64,
            rd,
            rm,
        } if rd.to_reg() == stack_reg() || rd.to_reg() == fp_reg() => {
            frame_copy = Some((rd.to_reg(), context.frame_offset(*rm)))
        }
        _ => {
            // `classify` is exhaustive. New memory instructions cannot become
            // silently pure merely because this precision table lacks an entry.
            if !legacy.steps.is_empty() {
                return Err(CodegenError::Unsupported(alloc::format!(
                    "AArch64 v4 memory coverage missing for {inst:?}"
                )));
            }
        }
    }
    // Every EA above uses the old physical register value. Only now may a
    // narrow W write, reload, fixed-register ABI instruction or scratch change
    // invalidate the old LocalBase and ABI frame facts.
    let mut defs = native_defs(inst);
    match inst {
        Inst::Call { info } => {
            call_pop_defs(info, &mut defs);
            if !info.patchable {
                retval_defs(info, frame, &mut defs);
            }
        }
        Inst::CallInd { info } => {
            call_pop_defs(info, &mut defs);
            retval_defs(info, frame, &mut defs);
        }
        Inst::ReturnCall { info } => tail_defs(info, frame, flags, &mut defs),
        Inst::ReturnCallInd { info } => tail_defs(info, frame, flags, &mut defs),
        _ => {}
    }
    if sp_after.is_some() {
        add_def(&mut defs, stack_reg());
    }
    materialization_defs(inst, frame, &mut defs);
    for reg in defs {
        seq.events.push(V4Event::RegisterDef(reg));
        context.def(reg);
    }
    if let Some(offset) = sp_after {
        context.set_frame_offset(stack_reg(), offset);
    }
    if let Some((reg, offset)) = frame_copy {
        context.set_frame_offset(reg, offset);
    }
    Ok(seq)
}

fn base_range(base: Reg, offset: i64, width: u32, context: &V4Context) -> V4Range {
    if base == stack_reg() || base == fp_reg() {
        // Once a frame identity is lost, do not infer it again from a reload or
        // an untrusted SP/FP value, even in the more precise range policies.
        context.frame_range(base, offset, width)
    } else {
        context.local_base(base, offset, width)
    }
}

fn normalized_offset(mem: &AMode, frame: &FrameLayout) -> Option<(Reg, i64)> {
    match mem {
        AMode::Unscaled { rn, simm9 } => Some((*rn, i64::from(simm9.value))),
        AMode::UnsignedOffset { rn, uimm12 } => Some((*rn, i64::from(uimm12.value()))),
        AMode::RegOffset { rn, off } => Some((*rn, *off)),
        AMode::SPOffset { off } => Some((stack_reg(), *off)),
        AMode::FPOffset { off } => Some((fp_reg(), *off)),
        AMode::SlotOffset { off } => off
            .checked_add(i64::from(frame.outgoing_args_size))
            .map(|off| (stack_reg(), off)),
        AMode::IncomingArg { off } => {
            let total = i64::from(frame.setup_area_size)
                + i64::from(frame.tail_args_size)
                + i64::from(frame.clobber_size)
                + i64::from(frame.fixed_frame_storage_size)
                + i64::from(frame.outgoing_args_size);
            total.checked_sub(*off).map(|off| (stack_reg(), off))
        }
        _ => None,
    }
}

fn mem_range(
    mem: &AMode,
    width: u32,
    frame: &FrameLayout,
    context: &V4Context,
    sp_after: &mut Option<Option<i64>>,
) -> V4Range {
    match mem {
        AMode::SPPreIndexed { simm9 } => {
            let delta = i64::from(simm9.value);
            *sp_after = Some(
                context
                    .frame_offset(stack_reg())
                    .and_then(|off| off.checked_add(delta)),
            );
            context.frame_range(stack_reg(), delta, width)
        }
        AMode::SPPostIndexed { simm9 } => {
            *sp_after = Some(
                context
                    .frame_offset(stack_reg())
                    .and_then(|off| off.checked_add(i64::from(simm9.value))),
            );
            context.frame_range(stack_reg(), 0, width)
        }
        _ => match normalized_offset(mem, frame) {
            Some((base, offset)) => {
                // Large pseudos materialize an extended register index. Their
                // stable ABI frame offset is still known, but general indexed
                // addresses are deliberately outside the LocalBase rule.
                if base != stack_reg()
                    && base != fp_reg()
                    && matches!(mem, AMode::RegOffset { .. })
                    && !encodable_offset(offset, width)
                {
                    V4Range::Unknown
                } else {
                    base_range(base, offset, width, context)
                }
            }
            None => V4Range::Unknown,
        },
    }
}

fn pair_range(
    mem: &PairAMode,
    width: u32,
    context: &V4Context,
    sp_after: &mut Option<Option<i64>>,
) -> V4Range {
    match mem {
        PairAMode::SignedOffset { reg, simm7 } => {
            base_range(*reg, i64::from(simm7.value), width, context)
        }
        PairAMode::SPPreIndexed { simm7 } => {
            let delta = i64::from(simm7.value);
            *sp_after = Some(
                context
                    .frame_offset(stack_reg())
                    .and_then(|off| off.checked_add(delta)),
            );
            context.frame_range(stack_reg(), delta, width)
        }
        PairAMode::SPPostIndexed { simm7 } => {
            *sp_after = Some(
                context
                    .frame_offset(stack_reg())
                    .and_then(|off| off.checked_add(i64::from(simm7.value))),
            );
            context.frame_range(stack_reg(), 0, width)
        }
    }
}

fn load(
    seq: &mut V4Seq,
    mem: &AMode,
    width: u32,
    frame: &FrameLayout,
    context: &V4Context,
    sp_after: &mut Option<Option<i64>>,
) {
    seq.events.push(V4Event::Load(mem_range(
        mem, width, frame, context, sp_after,
    )));
}
fn store(
    seq: &mut V4Seq,
    mem: &AMode,
    width: u32,
    frame: &FrameLayout,
    context: &V4Context,
    sp_after: &mut Option<Option<i64>>,
) {
    seq.events.push(V4Event::Store(mem_range(
        mem, width, frame, context, sp_after,
    )));
}

fn encodable_offset(offset: i64, width: u32) -> bool {
    (-256..=255).contains(&offset)
        || (offset >= 0 && offset % i64::from(width) == 0 && offset / i64::from(width) <= 4095)
}

fn add_def(defs: &mut SmallVec<[Reg; 8]>, reg: Reg) {
    // W/X and all scalar/vector aliases share their Cranelift Reg identity.
    if reg != super::zero_reg() && !reg.is_spillslot() && !defs.contains(&reg) {
        defs.push(reg);
    }
}

fn call_defs<T>(info: &CallInfo<T>, defs: &mut SmallVec<[Reg; 8]>) {
    for pair in &info.defs {
        if let RetLocation::Reg(preg, ..) = pair.location {
            add_def(defs, preg);
        }
        add_def(defs, pair.vreg.to_reg());
    }
    for preg in info.clobbers {
        add_def(defs, Reg::from(preg));
    }
    add_def(defs, link_reg());
}

fn retval_defs<T>(info: &CallInfo<T>, frame: &FrameLayout, defs: &mut SmallVec<[Reg; 8]>) {
    for op in call_retval_ops(info, frame.stackslots_size) {
        if let RetvalOp::Mem { inst, .. } = op {
            for reg in native_defs(&inst) {
                add_def(defs, reg);
            }
            materialization_defs(&inst, frame, defs);
        }
    }
}

fn call_pop_defs<T>(info: &CallInfo<T>, defs: &mut SmallVec<[Reg; 8]>) {
    if info.callee_pop_size > 0 {
        let size = i32::try_from(info.callee_pop_size).expect("callee popped more than 2GB");
        for inst in AArch64MachineDeps::gen_sp_reg_adjust(-size) {
            for reg in native_defs(&inst) {
                add_def(defs, reg);
            }
        }
    }
}

fn tail_defs<T>(
    info: &super::ReturnCallInfo<T>,
    frame: &FrameLayout,
    flags: &Flags,
    defs: &mut SmallVec<[Reg; 8]>,
) {
    for inst in return_call_setup_insts(frame, flags, info) {
        for reg in native_defs(&inst) {
            add_def(defs, reg);
        }
        if matches!(
            inst,
            Inst::LoadP64 {
                mem: PairAMode::SPPostIndexed { .. },
                ..
            } | Inst::FpuLoadP64 {
                mem: PairAMode::SPPostIndexed { .. },
                ..
            } | Inst::ULoad64 {
                mem: AMode::SPPostIndexed { .. },
                ..
            } | Inst::FpuLoad64 {
                mem: AMode::SPPostIndexed { .. },
                ..
            }
        ) {
            add_def(defs, stack_reg());
        }
        materialization_defs(&inst, frame, defs);
    }
}

/// Exhaustive physical write effects, including instructions emitted by ABI
/// and regalloc. OperandVisitor intentionally hides physical defs and cannot
/// replace this native table.
fn native_defs(inst: &Inst) -> SmallVec<[Reg; 8]> {
    let mut defs = SmallVec::new();
    match inst {
        Inst::AluRRR { rd, .. }
        | Inst::AluRRRR { rd, .. }
        | Inst::AluRRImm12 { rd, .. }
        | Inst::AluRRImmLogic { rd, .. }
        | Inst::AluRRImmShift { rd, .. }
        | Inst::AluRRRShift { rd, .. }
        | Inst::AluRRRExtend { rd, .. }
        | Inst::BitRR { rd, .. }
        | Inst::ULoad8 { rd, .. }
        | Inst::SLoad8 { rd, .. }
        | Inst::ULoad16 { rd, .. }
        | Inst::SLoad16 { rd, .. }
        | Inst::ULoad32 { rd, .. }
        | Inst::SLoad32 { rd, .. }
        | Inst::ULoad64 { rd, .. }
        | Inst::Mov { rd, .. }
        | Inst::MovFromPReg { rd, .. }
        | Inst::MovWide { rd, .. }
        | Inst::MovK { rd, .. }
        | Inst::Extend { rd, .. }
        | Inst::CSel { rd, .. }
        | Inst::CSNeg { rd, .. }
        | Inst::CSet { rd, .. }
        | Inst::CSetm { rd, .. }
        | Inst::AtomicCAS { rd, .. }
        | Inst::FpuMove32 { rd, .. }
        | Inst::FpuMove64 { rd, .. }
        | Inst::FpuMove128 { rd, .. }
        | Inst::FpuMoveFromVec { rd, .. }
        | Inst::FpuExtend { rd, .. }
        | Inst::FpuRR { rd, .. }
        | Inst::FpuRRR { rd, .. }
        | Inst::FpuRRI { rd, .. }
        | Inst::FpuRRIMod { rd, .. }
        | Inst::FpuRRRR { rd, .. }
        | Inst::FpuLoad16 { rd, .. }
        | Inst::FpuLoad32 { rd, .. }
        | Inst::FpuLoad64 { rd, .. }
        | Inst::FpuLoad128 { rd, .. }
        | Inst::FpuToInt { rd, .. }
        | Inst::IntToFpu { rd, .. }
        | Inst::FpuCSel16 { rd, .. }
        | Inst::FpuCSel32 { rd, .. }
        | Inst::FpuCSel64 { rd, .. }
        | Inst::FpuRound { rd, .. }
        | Inst::MovToFpu { rd, .. }
        | Inst::FpuMoveFPImm { rd, .. }
        | Inst::MovToVec { rd, .. }
        | Inst::MovFromVec { rd, .. }
        | Inst::MovFromVecSigned { rd, .. }
        | Inst::VecDup { rd, .. }
        | Inst::VecDupFromFpu { rd, .. }
        | Inst::VecDupFPImm { rd, .. }
        | Inst::VecDupImm { rd, .. }
        | Inst::VecExtend { rd, .. }
        | Inst::VecMovElement { rd, .. }
        | Inst::VecRRLong { rd, .. }
        | Inst::VecRRNarrowLow { rd, .. }
        | Inst::VecRRNarrowHigh { rd, .. }
        | Inst::VecRRPair { rd, .. }
        | Inst::VecRRRLong { rd, .. }
        | Inst::VecRRRLongMod { rd, .. }
        | Inst::VecRRPairLong { rd, .. }
        | Inst::VecRRR { rd, .. }
        | Inst::VecRRRMod { rd, .. }
        | Inst::VecFmlaElem { rd, .. }
        | Inst::VecMisc { rd, .. }
        | Inst::VecLanes { rd, .. }
        | Inst::VecShiftImm { rd, .. }
        | Inst::VecShiftImmMod { rd, .. }
        | Inst::VecExtract { rd, .. }
        | Inst::VecTbl { rd, .. }
        | Inst::VecTblExt { rd, .. }
        | Inst::VecTbl2 { rd, .. }
        | Inst::VecTbl2Ext { rd, .. }
        | Inst::VecLoadReplicate { rd, .. }
        | Inst::VecCSel { rd, .. }
        | Inst::MovFromNZCV { rd, .. }
        | Inst::Adr { rd, .. }
        | Inst::Adrp { rd, .. }
        | Inst::LoadExtNameGot { rd, .. }
        | Inst::LoadExtNameNear { rd, .. }
        | Inst::LoadExtNameFar { rd, .. }
        | Inst::LoadAddr { rd, .. } => add_def(&mut defs, rd.to_reg()),
        Inst::LoadP64 { rt, rt2, .. }
        | Inst::FpuLoadP64 { rt, rt2, .. }
        | Inst::FpuLoadP128 { rt, rt2, .. } => {
            add_def(&mut defs, rt.to_reg());
            add_def(&mut defs, rt2.to_reg());
        }
        Inst::AtomicRMW { rt, .. } | Inst::LoadAcquire { rt, .. } => {
            add_def(&mut defs, rt.to_reg())
        }
        Inst::AtomicRMWLoop {
            oldval,
            scratch1,
            scratch2,
            ..
        } => {
            for reg in [oldval.to_reg(), scratch1.to_reg(), scratch2.to_reg()] {
                add_def(&mut defs, reg);
            }
        }
        Inst::AtomicCASLoop {
            oldval, scratch, ..
        } => {
            add_def(&mut defs, oldval.to_reg());
            add_def(&mut defs, scratch.to_reg());
        }
        Inst::MovToPReg { rd, .. } => add_def(&mut defs, Reg::from(*rd)),
        Inst::Call { info } => call_defs(info, &mut defs),
        Inst::CallInd { info } => call_defs(info, &mut defs),
        Inst::Args { args } => {
            for pair in args {
                add_def(&mut defs, pair.preg);
            }
        }
        Inst::Paci { .. } | Inst::Xpaclri { .. } | Inst::AuthenticatedRet { .. } => {
            add_def(&mut defs, link_reg())
        }
        Inst::JTSequence { rtmp1, rtmp2, .. } => {
            add_def(&mut defs, rtmp1.to_reg());
            add_def(&mut defs, rtmp2.to_reg());
        }
        Inst::ElfTlsGetAddr { rd, tmp, .. } => {
            add_def(&mut defs, rd.to_reg());
            add_def(&mut defs, tmp.to_reg());
            add_def(&mut defs, link_reg());
        }
        Inst::MachOTlsGetAddr { rd, .. } => {
            add_def(&mut defs, rd.to_reg());
            for preg in AArch64MachineDeps::get_regs_clobbered_by_call(
                crate::isa::CallConv::AppleAarch64,
                false,
            ) {
                add_def(&mut defs, Reg::from(preg));
            }
            add_def(&mut defs, link_reg());
        }
        Inst::LabelAddress { dst, .. } => add_def(&mut defs, dst.to_reg()),
        Inst::StackProbeLoop { start, .. } => add_def(&mut defs, start.to_reg()),
        Inst::Nop0 { .. }
        | Inst::Nop4 { .. }
        | Inst::Store8 { .. }
        | Inst::Store16 { .. }
        | Inst::Store32 { .. }
        | Inst::Store64 { .. }
        | Inst::StoreP64 { .. }
        | Inst::CCmp { .. }
        | Inst::CCmpImm { .. }
        | Inst::StoreRelease { .. }
        | Inst::Fence { .. }
        | Inst::Csdb { .. }
        | Inst::Ssbb { .. }
        | Inst::FpuCmp { .. }
        | Inst::FpuStore16 { .. }
        | Inst::FpuStore32 { .. }
        | Inst::FpuStore64 { .. }
        | Inst::FpuStore128 { .. }
        | Inst::FpuStoreP64 { .. }
        | Inst::FpuStoreP128 { .. }
        | Inst::MovToNZCV { .. }
        | Inst::ReturnCall { .. }
        | Inst::ReturnCallInd { .. }
        | Inst::Rets { .. }
        | Inst::Ret { .. }
        | Inst::Jump { .. }
        | Inst::CondBr { .. }
        | Inst::TestBitAndBranch { .. }
        | Inst::TrapIf { .. }
        | Inst::IndirectBr { .. }
        | Inst::Brk { .. }
        | Inst::Udf { .. }
        | Inst::Word4 { .. }
        | Inst::Word8 { .. }
        | Inst::Bti { .. }
        | Inst::EmitIsland { .. }
        | Inst::Unwind { .. }
        | Inst::DummyUse { .. }
        | Inst::SequencePoint { .. } => {}
    }
    defs
}

/// The emitter can introduce address-constant scratch writes after regalloc.
/// Their old values are unusable for subsequent LocalBase proofs.
fn materialization_defs(inst: &Inst, frame: &FrameLayout, defs: &mut SmallVec<[Reg; 8]>) {
    let mem_width = match inst {
        Inst::ULoad8 { mem, .. } | Inst::SLoad8 { mem, .. } | Inst::Store8 { mem, .. } => {
            Some((mem, 1))
        }
        Inst::ULoad16 { mem, .. }
        | Inst::SLoad16 { mem, .. }
        | Inst::Store16 { mem, .. }
        | Inst::FpuLoad16 { mem, .. }
        | Inst::FpuStore16 { mem, .. } => Some((mem, 2)),
        Inst::ULoad32 { mem, .. }
        | Inst::SLoad32 { mem, .. }
        | Inst::Store32 { mem, .. }
        | Inst::FpuLoad32 { mem, .. }
        | Inst::FpuStore32 { mem, .. } => Some((mem, 4)),
        Inst::ULoad64 { mem, .. }
        | Inst::Store64 { mem, .. }
        | Inst::FpuLoad64 { mem, .. }
        | Inst::FpuStore64 { mem, .. } => Some((mem, 8)),
        Inst::FpuLoad128 { mem, .. } | Inst::FpuStore128 { mem, .. } => Some((mem, 16)),
        Inst::LoadAddr { mem, .. } => Some((mem, 1)),
        _ => None,
    };
    if let Some((mem, width)) = mem_width {
        if matches!(
            mem,
            AMode::RegOffset { .. }
                | AMode::SPOffset { .. }
                | AMode::FPOffset { .. }
                | AMode::SlotOffset { .. }
                | AMode::IncomingArg { .. }
        ) {
            if let Some((_, off)) = normalized_offset(mem, frame) {
                if !encodable_offset(off, width) {
                    add_def(defs, spilltmp_reg());
                }
            }
        }
        if matches!(inst, Inst::LoadAddr { .. }) {
            // An already concrete scaled offset is preserved by mem_finalize,
            // then LoadAddr may need a second scratch for its ADD. A large
            // pseudo offset instead becomes RegExtended and needs only x16.
            if let AMode::UnsignedOffset { uimm12, .. } = mem {
                if super::Imm12::maybe_from_u64(u64::from(uimm12.value())).is_none() {
                    add_def(defs, tmp2_reg());
                }
            }
        }
    }
}

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
