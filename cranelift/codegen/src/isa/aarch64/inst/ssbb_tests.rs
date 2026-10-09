use super::{RetvalOp, SSBB_ENCODING, call_retval_ops, classify};
use crate::cursor::{Cursor, FuncCursor};
use crate::ir::condcodes::IntCC;
use crate::ir::{
    AbiParam, AtomicRmwOp, BlockArg, ExtFuncData, ExternalName, FuncRef, Function, InstBuilder,
    JumpTableData, MemFlags, Signature, TrapCode, Type, UserExternalName, UserFuncName, Value,
    types,
};
use crate::isa::aarch64::inst::emit::enc_ldst_imm19;
use crate::isa::aarch64::inst::{Inst, LabelUse, spilltmp_reg, writable_xreg, xreg};
use crate::isa::{self, CallConv};
use crate::machinst::ssbb::{SsbbMemStep, SsbbSeq};
use crate::machinst::{FrameLayout, MachBuffer};
use crate::settings::{self, Configurable};
use crate::{CodegenError, Context};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use cranelift_control::ControlPlane;

const SSBB: u32 = SSBB_ENCODING;
const CSDB: u32 = 0xD503229F;

fn selective_seq(
    inst: &Inst,
    context: &mut crate::machinst::v4_fences::V4Context,
) -> crate::machinst::v4_fences::V4Seq {
    super::classify_v4(
        inst,
        &FrameLayout::default(),
        &settings::Flags::new(settings::builder()),
        context,
    )
    .unwrap()
}

#[test]
fn selective_narrow_vector_pair_and_atomic_widths() {
    use crate::isa::aarch64::inst::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy, V4Range};
    let flags = MemFlags::trusted();
    let mut context = V4Context::new(7, V4Policy::Ranges);
    let mem = AMode::reg(xreg(0));
    for (inst, width) in [
        (
            Inst::SLoad8 {
                rd: writable_xreg(1),
                mem: mem.clone(),
                flags,
            },
            1,
        ),
        (
            Inst::SLoad16 {
                rd: writable_xreg(1),
                mem: mem.clone(),
                flags,
            },
            2,
        ),
        (
            Inst::SLoad32 {
                rd: writable_xreg(1),
                mem: mem.clone(),
                flags,
            },
            4,
        ),
        (
            Inst::ULoad64 {
                rd: writable_xreg(1),
                mem: mem.clone(),
                flags,
            },
            8,
        ),
        (
            Inst::FpuLoad128 {
                rd: writable_vreg(1),
                mem,
                flags,
            },
            16,
        ),
        (
            Inst::VecLoadReplicate {
                rd: writable_vreg(1),
                rn: xreg(0),
                size: VectorSize::Size16x8,
                flags,
            },
            2,
        ),
        (
            Inst::LoadP64 {
                rt: writable_xreg(1),
                rt2: writable_xreg(2),
                mem: PairAMode::SignedOffset {
                    reg: xreg(0),
                    simm7: SImm7Scaled::maybe_from_i64(16, types::I64).unwrap(),
                },
                flags,
            },
            16,
        ),
        (
            Inst::FpuLoadP128 {
                rt: writable_vreg(1),
                rt2: writable_vreg(2),
                mem: PairAMode::SignedOffset {
                    reg: xreg(0),
                    simm7: SImm7Scaled::maybe_from_i64(16, types::I8X16).unwrap(),
                },
                flags,
            },
            32,
        ),
    ] {
        let seq = selective_seq(&inst, &mut context);
        assert!(
            matches!(seq.events[0], V4Event::Load(V4Range::LocalBase { width: actual, .. }) if actual == width),
            "{inst:?}: {seq:?}"
        );
    }
    for ty in [types::I8, types::I16, types::I32, types::I64] {
        let inst = Inst::AtomicCAS {
            rd: writable_xreg(1),
            rs: xreg(1),
            rt: xreg(2),
            rn: xreg(0),
            ty,
            flags,
        };
        let seq = selective_seq(&inst, &mut context);
        assert!(
            matches!(seq.events[0], V4Event::Rmw(V4Range::LocalBase { width, .. }, V4Range::LocalBase { width: write, .. }) if width == ty.bytes() && write == width)
        );
    }
    let mem = AMode::reg(xreg(0));
    for (inst, width) in [
        (
            Inst::Store8 {
                rd: xreg(1),
                mem: mem.clone(),
                flags,
            },
            1,
        ),
        (
            Inst::Store16 {
                rd: xreg(1),
                mem: mem.clone(),
                flags,
            },
            2,
        ),
        (
            Inst::Store32 {
                rd: xreg(1),
                mem: mem.clone(),
                flags,
            },
            4,
        ),
        (
            Inst::Store64 {
                rd: xreg(1),
                mem: mem.clone(),
                flags,
            },
            8,
        ),
        (
            Inst::FpuStore128 {
                rd: vreg(1),
                mem,
                flags,
            },
            16,
        ),
        (
            Inst::FpuStoreP128 {
                rt: vreg(1),
                rt2: vreg(2),
                mem: PairAMode::SignedOffset {
                    reg: xreg(0),
                    simm7: SImm7Scaled::maybe_from_i64(0, types::I8X16).unwrap(),
                },
                flags,
            },
            32,
        ),
    ] {
        let seq = selective_seq(&inst, &mut context);
        assert!(
            matches!(seq.events[0], V4Event::Store(V4Range::LocalBase { width: actual, .. }) if actual == width),
            "{inst:?}: {seq:?}"
        );
    }
}

#[test]
fn selective_writeback_uses_old_address_then_invalidates_loaded_fp() {
    use crate::isa::aarch64::inst::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy, V4Range};
    let mut context = V4Context::new(0, V4Policy::Frame);
    context.set_frame_offset(stack_reg(), Some(0));
    let push = Inst::StoreP64 {
        rt: fp_reg(),
        rt2: link_reg(),
        mem: PairAMode::SPPreIndexed {
            simm7: SImm7Scaled::maybe_from_i64(-16, types::I64).unwrap(),
        },
        flags: MemFlags::trusted(),
    };
    assert_eq!(
        selective_seq(&push, &mut context).events[0],
        V4Event::Store(V4Range::Frame {
            frame: 0,
            offset: -16,
            width: 16
        })
    );
    assert_eq!(context.frame_offset(stack_reg()), Some(-16));
    let bind = Inst::AluRRImm12 {
        alu_op: ALUOp::Add,
        size: OperandSize::Size64,
        rd: writable_fp_reg(),
        rn: stack_reg(),
        imm12: Imm12::ZERO,
    };
    selective_seq(&bind, &mut context);
    assert_eq!(context.frame_offset(fp_reg()), Some(-16));
    let pop = Inst::LoadP64 {
        rt: writable_fp_reg(),
        rt2: writable_link_reg(),
        mem: PairAMode::SPPostIndexed {
            simm7: SImm7Scaled::maybe_from_i64(16, types::I64).unwrap(),
        },
        flags: MemFlags::trusted(),
    };
    let seq = selective_seq(&pop, &mut context);
    assert_eq!(
        seq.events[0],
        V4Event::Load(V4Range::Frame {
            frame: 0,
            offset: -16,
            width: 16
        })
    );
    assert!(seq.events.contains(&V4Event::RegisterDef(fp_reg())));
    assert_eq!(context.frame_offset(stack_reg()), Some(0));
    assert_eq!(context.frame_offset(fp_reg()), None);
}

#[test]
fn selective_old_effective_address_precedes_w_alias_definition() {
    use crate::isa::aarch64::inst::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy, V4Range};
    let mut context = V4Context::new(3, V4Policy::Ranges);
    let load = Inst::ULoad32 {
        rd: writable_xreg(0),
        mem: AMode::reg(xreg(0)),
        flags: MemFlags::trusted(),
    };
    let seq = selective_seq(&load, &mut context);
    assert_eq!(
        seq.events[0],
        V4Event::Load(V4Range::LocalBase {
            block: 3,
            reg: xreg(0),
            epoch: 0,
            offset: 0,
            width: 4
        })
    );
    assert!(seq.events[1..].contains(&V4Event::RegisterDef(xreg(0))));
    assert_eq!(
        context.local_base(xreg(0), 0, 8),
        V4Range::LocalBase {
            block: 3,
            reg: xreg(0),
            epoch: 1,
            offset: 0,
            width: 8
        }
    );
    for mem in [
        AMode::RegReg {
            rn: xreg(0),
            rm: xreg(2),
        },
        AMode::RegScaled {
            rn: xreg(0),
            rm: xreg(2),
        },
        AMode::RegExtended {
            rn: xreg(0),
            rm: xreg(2),
            extendop: ExtendOp::UXTW,
        },
    ] {
        let seq = selective_seq(
            &Inst::ULoad64 {
                rd: writable_xreg(1),
                mem,
                flags: MemFlags::trusted(),
            },
            &mut context,
        );
        assert_eq!(seq.events[0], V4Event::Load(V4Range::Unknown));
    }
}

#[test]
fn selective_single_simd_writeback_and_large_frame_offsets() {
    use crate::isa::aarch64::inst::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy, V4Range};
    let mut context = V4Context::new(0, V4Policy::Frame);
    context.set_frame_offset(stack_reg(), Some(-128));
    let inst = Inst::FpuLoad64 {
        rd: writable_vreg(0),
        mem: AMode::SPPostIndexed {
            simm9: SImm9::maybe_from_i64(16).unwrap(),
        },
        flags: MemFlags::trusted(),
    };
    assert_eq!(
        selective_seq(&inst, &mut context).events[0],
        V4Event::Load(V4Range::Frame {
            frame: 0,
            offset: -128,
            width: 8
        })
    );
    assert_eq!(context.frame_offset(stack_reg()), Some(-112));
    let mut frame = FrameLayout::default();
    frame.outgoing_args_size = 64;
    let inst = Inst::Store64 {
        rd: xreg(0),
        mem: AMode::SlotOffset { off: 40_000 },
        flags: MemFlags::trusted(),
    };
    let seq = super::classify_v4(
        &inst,
        &frame,
        &settings::Flags::new(settings::builder()),
        &mut context,
    )
    .unwrap();
    assert_eq!(
        seq.events[0],
        V4Event::Store(V4Range::Frame {
            frame: 0,
            offset: 39_952,
            width: 8
        })
    );
    assert!(seq.events.contains(&V4Event::RegisterDef(spilltmp_reg())));
    let inst = Inst::ULoad64 {
        rd: writable_xreg(0),
        mem: AMode::RegOffset {
            rn: xreg(1),
            off: 40_000,
        },
        flags: MemFlags::trusted(),
    };
    let seq = selective_seq(&inst, &mut context);
    assert_eq!(seq.events[0], V4Event::Load(V4Range::Unknown));
}

fn compile_policy(policy: &str, func: Function) -> Compiled {
    let mut shared = settings::builder();
    shared.set("opt_level", "none").unwrap();
    shared.set("v4_fence_policy", policy).unwrap();
    let mut builder = isa::lookup("aarch64".parse().unwrap()).unwrap();
    builder.set("has_lse", "true").unwrap();
    builder.set("aarch64_mitigation", "ssbb").unwrap();
    let isa = builder.finish(settings::Flags::new(shared)).unwrap();
    let mut ctx = Context::for_function(func);
    ctx.set_disasm(true);
    match ctx.compile(&*isa, &mut ControlPlane::default()) {
        Ok(code) => Compiled {
            bytes: code.buffer.data().to_vec(),
            disasm: code.vcode.clone().unwrap_or_default(),
            traps: code.buffer.traps().len(),
        },
        Err(err) => panic!("compile {policy} failed: {:?}", err.inner),
    }
}

/// Exercise actual classifier, planner and emitter together, without VCode or
/// frontend transformations hiding a redundant template-entry barrier.
fn selective_boundary_native_code(insts: &[Inst]) -> (Vec<u8>, Vec<bool>, u32) {
    use crate::isa::aarch64::inst::{EmitInfo, EmitState};
    use crate::machinst::v4_fences::{
        self, V4Block, V4Context, V4EventId, V4Graph, V4Node, V4Policy,
    };
    use crate::machinst::{MachInstEmit, MachInstEmitState};
    let flags = settings::Flags::new(settings::builder());
    let mut context = V4Context::new(0, V4Policy::Ranges);
    let mut events = Vec::new();
    let mut sequences = Vec::new();
    for (site, inst) in insts.iter().enumerate() {
        let seq = super::classify_v4(inst, &FrameLayout::default(), &flags, &mut context).unwrap();
        for (sub, event) in seq.events.iter().copied().enumerate() {
            events.push(V4Node {
                id: V4EventId {
                    block: 0,
                    inst: site as u32,
                    sub: sub as u16,
                },
                event,
                guard: Some(site),
            });
        }
        sequences.push(seq);
    }
    let graph = V4Graph {
        blocks: vec![V4Block {
            events,
            successors: vec![],
            exit_site: None,
            loop_depth: 0,
        }],
        entry: 0,
        sites: insts.len(),
    };
    let plan = v4_fences::plan(&graph, V4Policy::Ranges).unwrap();
    assert!(v4_fences::check(&graph, &plan.fences).errors.is_empty());
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.v4_set_selective(true);
    let emit_info = EmitInfo::new(flags);
    let mut sink = MachBuffer::new();
    for (site, (inst, seq)) in insts.iter().zip(&sequences).enumerate() {
        if plan.fences[site] {
            state.ssbb_emit_barrier(&mut sink, &emit_info);
        }
        state.v4_set_legacy_internal(seq.legacy_internal);
        inst.emit(&mut sink, &emit_info, &mut state);
        state.v4_set_legacy_internal(false);
    }
    (
        sink.testing_data().to_vec(),
        plan.fences,
        sequences.iter().map(|seq| seq.fixed_barriers()).sum(),
    )
}

fn disjoint_range_load(alias_after: bool) -> Function {
    let mut func = new_func(&[types::I64, types::I64], &[types::I64]);
    let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
    let first = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 32);
    pos.ins().store(MemFlags::trusted(), vals[1], vals[0], 0);
    let unrelated = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 16);
    let mut sum = pos.ins().iadd(first, unrelated);
    if alias_after {
        let related = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 0);
        sum = pos.ins().iadd(sum, related);
    }
    pos.ins().return_(&[sum]);
    func
}

#[test]
fn selective_native_codegen_moves_guard_and_keeps_nonfence_words() {
    let legacy = compile_policy("legacy", disjoint_range_load(false));
    let default = must_compile("ssbb", true, false, disjoint_range_load(false));
    assert_eq!(
        legacy.bytes, default.bytes,
        "legacy must preserve default emission"
    );
    assert_eq!(count_word(&legacy.bytes, SSBB), 2, "{}", legacy.disasm);
    for policy in ["frame", "ranges", "ranges_cfg"] {
        let code = compile_policy(policy, disjoint_range_load(false));
        assert_eq!(
            count_word(&code.bytes, SSBB),
            if policy == "frame" { 2 } else { 1 },
            "{policy}: {}",
            code.disasm
        );
        let nonfence = |code: &Compiled| {
            words(&code.bytes)
                .into_iter()
                .filter(|word| *word != SSBB)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            nonfence(&code),
            nonfence(&legacy),
            "{policy} changed original native instructions"
        );
    }
    let code = compile_policy("ranges", disjoint_range_load(true));
    assert_eq!(count_word(&code.bytes, SSBB), 2, "{}", code.disasm);
    // The second guard must follow the unrelated load, not precede it.
    let lines: Vec<_> = code.disasm.lines().map(str::trim).collect();
    let second_fence = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("dsb #0"))
        .nth(1)
        .unwrap()
        .0;
    assert!(
        lines[..second_fence]
            .iter()
            .any(|line| line.starts_with("ldr ") && line.contains("#16")),
        "{}",
        code.disasm
    );
}

#[test]
#[cfg(feature = "disas")]
fn selective_cfg_layout_preserves_folded_empty_retry_edge() {
    use crate::machinst::v4_fences::decoded_nonfence_stream;
    let mut func = new_func(&[types::I64], &[types::I64]);
    let (mut pos, values) = entry(&mut func, &[types::I64]);
    let ptr = values[0];
    let desired = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 8);
    let retry = pos.func.dfg.make_block();
    let failed = pos.func.dfg.make_block();
    let done = pos.func.dfg.make_block();
    pos.ins().jump(retry, no_args());
    pos.insert_block(retry);
    let expected = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
    let old = pos
        .ins()
        .atomic_cas(MemFlags::trusted(), ptr, expected, desired);
    let success = pos.ins().icmp(IntCC::Equal, old, expected);
    pos.ins().brif(success, done, no_args(), failed, no_args());
    pos.insert_block(failed);
    pos.ins().jump(retry, no_args());
    pos.insert_block(done);
    pos.ins().return_(&[old]);
    let cs = target("ssbb", true, false).to_capstone().unwrap();
    let reference = compile_policy("ranges", func.clone());
    let optimized = compile_policy("ranges_cfg", func);
    assert_eq!(
        decoded_nonfence_stream(&cs, &optimized.bytes, &SSBB.to_le_bytes()),
        decoded_nonfence_stream(&cs, &reference.bytes, &SSBB.to_le_bytes()),
        "an empty retry edge must remain eligible for native branch folding"
    );
}

#[test]
fn selective_cfg_layout_separator_rejects_zero_code_and_complex_tails() {
    use crate::isa::aarch64::inst::*;
    use crate::machinst::MachInstEmit;
    let ordinary = Inst::Store64 {
        rd: xreg(1),
        mem: AMode::reg(xreg(0)),
        flags: MemFlags::trusted(),
    };
    assert!(ordinary.v4_emits_nonbranch_code());
    let paired = Inst::StoreP64 {
        rt: xreg(1),
        rt2: xreg(2),
        mem: PairAMode::SignedOffset {
            reg: xreg(0),
            simm7: SImm7Scaled::maybe_from_i64(0, types::I64).unwrap(),
        },
        flags: MemFlags::trusted(),
    };
    assert!(paired.v4_emits_nonbranch_code());
    for inst in [
        Inst::Args { args: vec![] },
        Inst::StackProbeLoop {
            start: writable_xreg(0),
            end: xreg(1),
            step: Imm12::maybe_from_u64(4096).unwrap(),
        },
        Inst::AtomicRMW {
            ty: types::I64,
            op: AtomicRMWOp::Add,
            rs: xreg(1),
            rt: writable_xreg(2),
            rn: xreg(0),
            flags: MemFlags::trusted(),
        },
    ] {
        assert!(!inst.v4_emits_nonbranch_code(), "{inst:?}");
    }
}

#[test]
#[cfg(feature = "disas")]
fn selective_cfg_layout_moves_guard_after_real_store() {
    use crate::machinst::v4_fences::decoded_nonfence_stream;
    let mut func = new_func(&[types::I64, types::I32], &[types::I64]);
    let (mut pos, values) = entry(&mut func, &[types::I64, types::I32]);
    let seed = pos
        .ins()
        .load(types::I64, MemFlags::trusted(), values[0], 16);
    let dirty = pos.func.dfg.make_block();
    let clean = pos.func.dfg.make_block();
    let join = pos.func.dfg.make_block();
    let condition = pos.ins().icmp_imm(IntCC::NotEqual, values[1], 0);
    pos.ins()
        .brif(condition, dirty, no_args(), clean, no_args());
    pos.insert_block(dirty);
    pos.ins().store(MemFlags::trusted(), seed, values[0], 0);
    pos.ins().jump(join, no_args());
    pos.insert_block(clean);
    pos.ins().jump(join, no_args());
    pos.insert_block(join);
    let result = pos
        .ins()
        .load(types::I64, MemFlags::trusted(), values[0], 0);
    pos.ins().return_(&[result]);
    let cs = target("ssbb", true, false).to_capstone().unwrap();
    let reference = compile_policy("ranges", func.clone());
    let optimized = compile_policy("ranges_cfg", func);
    assert_ne!(
        optimized.bytes, reference.bytes,
        "a store followed by a jump must allow CFG placement"
    );
    assert_eq!(
        decoded_nonfence_stream(&cs, &optimized.bytes, &SSBB.to_le_bytes()),
        decoded_nonfence_stream(&cs, &reference.bytes, &SSBB.to_le_bytes()),
    );
    let decoded = cs.disasm_all(&optimized.bytes, 0).unwrap();
    assert!(
        decoded.as_ref().windows(2).any(|pair| {
            pair[0].mnemonic() == Some("str") && pair[1].bytes() == SSBB.to_le_bytes()
        }),
        "the moved guard must follow the actual store"
    );
}

fn frozen_boundary_templates() -> Vec<Inst> {
    use crate::isa::aarch64::inst::*;
    let symbol = alloc::boxed::Box::new(ExternalName::testcase("v4"));
    vec![
        Inst::LoadExtNameGot {
            rd: writable_xreg(0),
            name: symbol.clone(),
        },
        Inst::LoadExtNameFar {
            rd: writable_xreg(0),
            name: symbol.clone(),
            offset: 0,
        },
        Inst::ElfTlsGetAddr {
            rd: writable_xreg(0),
            tmp: writable_xreg(1),
            symbol: symbol.clone(),
        },
        Inst::MachOTlsGetAddr {
            rd: writable_xreg(0),
            symbol: *symbol,
        },
        Inst::ReturnCallInd {
            info: alloc::boxed::Box::new(ReturnCallInfo {
                dest: xreg(1),
                uses: smallvec::smallvec![],
                new_stack_arg_size: 0,
                key: None,
            }),
        },
    ]
}

#[test]
fn selective_frozen_template_entry_guards_follow_pending() {
    use crate::isa::aarch64::inst::*;
    use crate::machinst::MachInstEmitState;
    use crate::machinst::v4_fences::{V4Context, V4Policy};
    let flags = settings::Flags::new(settings::builder());
    for inst in frozen_boundary_templates() {
        let mut context = V4Context::new(0, V4Policy::Ranges);
        let seq = super::classify_v4(&inst, &FrameLayout::default(), &flags, &mut context).unwrap();
        assert!(seq.legacy_internal, "{inst:?}");
        assert_eq!(
            seq.fixed_barriers(),
            0,
            "a template's first read is an external protection site: {inst:?}"
        );
        let clean = [Inst::Ssbb {}, inst.clone()];
        let (bytes, guards, fixed) = selective_boundary_native_code(&clean);
        assert_eq!(guards, vec![false, false], "clean template entry: {inst:?}");
        assert_eq!(fixed, 0, "{inst:?}");
        assert_eq!(
            count_word(&bytes, SSBB),
            1,
            "only the explicit upstream fence should remain: {inst:?}"
        );
        let dirty = [
            Inst::Store64 {
                rd: xreg(1),
                mem: AMode::reg(xreg(2)),
                flags: MemFlags::trusted(),
            },
            inst.clone(),
        ];
        let (bytes, guards, fixed) = selective_boundary_native_code(&dirty);
        assert_eq!(
            guards,
            vec![false, true],
            "dirty template entry must be guarded by the plan: {inst:?}"
        );
        assert_eq!(fixed, 0, "{inst:?}");
        assert_eq!(
            count_word(&bytes, SSBB),
            1,
            "dirty entry gets exactly one external fence: {inst:?}"
        );
    }
    // Selective ordinary emission must not reinsert the legacy automatic guard.
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.v4_set_selective(true);
    state.ssbb_set_dirty(true);
    let mut sink = MachBuffer::new();
    state.ssbb_step(SsbbMemStep::Load, &mut sink);
    assert!(sink.testing_data().is_empty());
}

#[test]
fn selective_frozen_emitter_does_not_force_entry_guard() {
    use crate::isa::aarch64::inst::{EmitInfo, EmitState};
    use crate::machinst::{MachInstEmit, MachInstEmitState};
    let info = EmitInfo::new(settings::Flags::new(settings::builder()));
    for inst in frozen_boundary_templates() {
        for previous_dirty in [false, true] {
            let mut state = EmitState::default();
            state.ssbb_install(&[]);
            state.v4_set_selective(true);
            state.ssbb_set_dirty(previous_dirty);
            // Input Pending belongs to the planner. The internal emitter must
            // start clean even if its previous lexical cursor was dirty.
            state.v4_set_legacy_internal(true);
            let mut sink = MachBuffer::new();
            inst.emit(&mut sink, &info, &mut state);
            assert_eq!(
                count_word(sink.testing_data(), SSBB),
                0,
                "template emitted an unconditional entry guard: {inst:?}, previous dirty={previous_dirty}"
            );
            assert_eq!(state.ssbb_barrier_count(), 0);
        }
    }
}

#[test]
#[cfg(feature = "disas")]
fn selective_frozen_literal_tls_and_tail_native_boundaries() {
    use crate::isa::aarch64::inst::{AMode, xreg};
    let cs = target("ssbb", true, false).to_capstone().unwrap();
    for template in frozen_boundary_templates() {
        // The far-literal template ends with an eight-byte relocation payload.
        // Audit those bytes as data, never as fabricated native instructions.
        let literal_size = if matches!(template, Inst::LoadExtNameFar { .. }) {
            8
        } else {
            0
        };
        let clean = [Inst::Ssbb {}, template.clone()];
        let (clean_bytes, _, _) = selective_boundary_native_code(&clean);
        let clean_code_end = clean_bytes.len() - literal_size;
        let clean_decoded = cs.disasm_all(&clean_bytes[..clean_code_end], 0).unwrap();
        assert_eq!(
            clean_decoded
                .iter()
                .map(|inst| inst.bytes().len())
                .sum::<usize>(),
            clean_code_end,
            "undecoded template code: {template:?}"
        );
        assert_eq!(
            clean_decoded
                .iter()
                .filter(|inst| inst.bytes() == SSBB.to_le_bytes())
                .count(),
            1,
            "{template:?}"
        );
        let dirty = [
            Inst::Store64 {
                rd: xreg(1),
                mem: AMode::reg(xreg(2)),
                flags: MemFlags::trusted(),
            },
            template.clone(),
        ];
        let (dirty_bytes, _, _) = selective_boundary_native_code(&dirty);
        let dirty_code_end = dirty_bytes.len() - literal_size;
        let dirty_decoded = cs.disasm_all(&dirty_bytes[..dirty_code_end], 0).unwrap();
        assert_eq!(
            dirty_decoded
                .iter()
                .map(|inst| inst.bytes().len())
                .sum::<usize>(),
            dirty_code_end,
            "undecoded template code: {template:?}"
        );
        for instruction in clean_decoded.iter().chain(dirty_decoded.iter()) {
            cs.insn_detail(instruction)
                .expect("actual template code must decode as instructions, without skip-data");
        }
        let words = dirty_decoded.as_ref();
        assert_eq!(words[0].mnemonic(), Some("str"));
        assert_eq!(
            words[1].bytes(),
            SSBB.to_le_bytes(),
            "dirty entry must be protected before the template: {template:?}"
        );
        assert_eq!(
            words
                .iter()
                .filter(|inst| inst.bytes() == SSBB.to_le_bytes())
                .count(),
            1,
            "{template:?}"
        );
        // Removing only the dirty prefix store must leave the exact same native
        // template operations and their branch/literal layout.
        let dirty_template = &dirty_bytes[8..dirty_code_end];
        let clean_template = &clean_bytes[4..clean_code_end];
        assert_eq!(dirty_template, clean_template, "{template:?}");
        assert_eq!(
            &dirty_bytes[dirty_code_end..],
            &clean_bytes[clean_code_end..],
            "literal payload changed: {template:?}"
        );
        if literal_size != 0 {
            assert_eq!(
                &clean_bytes[clean_code_end..],
                &[0; 8],
                "the unresolved literal relocation keeps its payload placeholder"
            );
        }
    }
}

#[test]
fn selective_clean_opaque_call_needs_no_extra_guard_but_dirty_call_does() {
    use crate::isa::aarch64::inst::{AMode, OperandSize};
    use crate::machinst::abi::CallInfo;
    let load = Inst::ULoad64 {
        rd: writable_xreg(1),
        mem: AMode::reg(xreg(0)),
        flags: MemFlags::trusted(),
    };
    let pure = Inst::Mov {
        size: OperandSize::Size64,
        rd: writable_xreg(3),
        rm: xreg(1),
    };
    let call = Inst::CallInd {
        info: alloc::boxed::Box::new(CallInfo::empty(xreg(2), CallConv::SystemV)),
    };
    let (bytes, guards, fixed) =
        selective_boundary_native_code(&[Inst::Ssbb {}, load, pure, call.clone()]);
    assert_eq!(guards, vec![false, false, false, false]);
    assert_eq!(fixed, 0);
    assert_eq!(count_word(&bytes, SSBB), 1);
    let store = Inst::Store64 {
        rd: xreg(1),
        mem: AMode::reg(xreg(0)),
        flags: MemFlags::trusted(),
    };
    let (bytes, guards, fixed) = selective_boundary_native_code(&[store, call]);
    assert_eq!(guards, vec![false, true]);
    assert_eq!(fixed, 0);
    assert_eq!(count_word(&bytes, SSBB), 1);
    assert_eq!(words(&bytes)[1], SSBB);
}

#[test]
fn selective_lse_read_modify_write_keeps_post_write_load_protection() {
    use crate::isa::aarch64::inst::AMode;
    let load = Inst::ULoad64 {
        rd: writable_xreg(4),
        mem: AMode::reg(xreg(0)),
        flags: MemFlags::trusted(),
    };
    let atomic = Inst::AtomicCAS {
        rd: writable_xreg(1),
        rs: xreg(1),
        rt: xreg(2),
        rn: xreg(0),
        ty: types::I64,
        flags: MemFlags::trusted(),
    };
    let (bytes, guards, fixed) =
        selective_boundary_native_code(&[Inst::Ssbb {}, load.clone(), atomic, load]);
    assert_eq!(guards, vec![false, false, false, true]);
    assert_eq!(fixed, 0);
    assert_eq!(count_word(&bytes, SSBB), 2);
    assert!(
        words(&bytes)
            .windows(2)
            .any(|pair| is_cas(pair[0]) && pair[1] == SSBB),
        "CAS writes must guard the following aliasing load"
    );
}

#[test]
fn selective_try_call_cut_precedes_spilled_return_loads() {
    use crate::isa::aarch64::inst::{EmitInfo, EmitState};
    use crate::machinst::abi::{CallInfo, CallRetPair, RetLocation, StackAMode, TryCallInfo};
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy};
    use crate::machinst::{MachInstEmit, MachInstEmitState, Reg, Writable};
    let mut sink = MachBuffer::<Inst>::new();
    let continuation = sink.get_label();
    let mut info = CallInfo::empty(xreg(0), CallConv::SystemV);
    for i in 0..2 {
        info.defs.push(CallRetPair {
            vreg: Writable::from_reg(Reg::from(regalloc2::SpillSlot::new(i))),
            location: RetLocation::Stack(StackAMode::Slot((i * 8) as i64), types::I64),
        });
    }
    info.try_call_info = Some(TryCallInfo {
        continuation,
        exception_handlers: alloc::boxed::Box::new([]),
    });
    let inst = Inst::CallInd {
        info: alloc::boxed::Box::new(info),
    };
    let flags = settings::Flags::new(settings::builder());
    let mut context = V4Context::new(0, V4Policy::Ranges);
    let seq = super::classify_v4(&inst, &FrameLayout::default(), &flags, &mut context).unwrap();
    assert_eq!(seq.exn_cut, Some(0));
    assert_eq!(seq.events[seq.exn_cut.unwrap()], V4Event::OpaqueCall);
    assert_eq!(
        seq.fixed_barriers(),
        2,
        "only the two call/store-to-return-load transitions are fixed: {seq:?}"
    );
    assert!(
        seq.events
            .iter()
            .skip(2)
            .any(|event| matches!(event, V4Event::Store(_)))
    );
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.v4_set_selective(true);
    state.v4_set_legacy_internal(true);
    inst.emit(&mut sink, &EmitInfo::new(flags), &mut state);
    assert_eq!(
        count_word(sink.testing_data(), SSBB),
        2,
        "a clean call entry still protects both spilled return loads"
    );
}

#[test]
fn selective_fixed_native_stack_ranges_reduce_guard_count() {
    let make_func = || {
        let mut func = new_func(&[types::I64], &[types::I64]);
        let slot = func.create_sized_stack_slot(crate::ir::StackSlotData::new(
            crate::ir::StackSlotKind::ExplicitSlot,
            64,
            3,
        ));
        let (mut pos, vals) = entry(&mut func, &[types::I64]);
        let first = pos.ins().stack_load(types::I64, slot, 32);
        pos.ins().stack_store(vals[0], slot, 0);
        let unrelated = pos.ins().stack_load(types::I64, slot, 16);
        let sum = pos.ins().iadd(first, unrelated);
        pos.ins().return_(&[sum]);
        func
    };
    let legacy = compile_policy("legacy", make_func());
    let optimized = compile_policy("frame", make_func());
    assert!(
        count_word(&optimized.bytes, SSBB) < count_word(&legacy.bytes, SSBB),
        "legacy:\n{}\nframe:\n{}",
        legacy.disasm,
        optimized.disasm
    );
    let nonfence = |code: &Compiled| {
        words(&code.bytes)
            .into_iter()
            .filter(|word| *word != SSBB)
            .collect::<Vec<_>>()
    };
    assert_eq!(nonfence(&legacy), nonfence(&optimized));
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn count_word(bytes: &[u8], word: u32) -> usize {
    words(bytes)
        .into_iter()
        .filter(|item| *item == word)
        .count()
}

fn is_cas(word: u32) -> bool {
    const MASK: u32 = 0x3FE0FC00;
    const BITS: u32 = 0x08E0FC00;
    word & MASK == BITS
}

fn is_adr16(word: u32) -> bool {
    word & !0x1f == 0x10000080
}

fn step_kind(mnemonic: &str) -> &'static str {
    if mnemonic == "dsb" {
        return "barrier";
    }
    if mnemonic == "csdb" || mnemonic == "dmb" {
        return "neutral";
    }
    if mnemonic == "bl" || mnemonic == "blr" || mnemonic.starts_with("return_call") {
        return "call";
    }
    if mnemonic.starts_with("cas")
        || mnemonic.starts_with("ldadd")
        || mnemonic.starts_with("ldclr")
        || mnemonic.starts_with("ldeor")
        || mnemonic.starts_with("ldset")
        || mnemonic.starts_with("ldsmax")
        || mnemonic.starts_with("ldumax")
        || mnemonic.starts_with("ldsmin")
        || mnemonic.starts_with("ldumin")
        || mnemonic.starts_with("swp")
    {
        return "atomic";
    }
    if mnemonic.starts_with("ldr")
        || mnemonic.starts_with("ldp")
        || mnemonic.starts_with("ldur")
        || mnemonic.starts_with("ldar")
        || mnemonic.starts_with("ld1")
    {
        return "load";
    }
    if mnemonic.starts_with("str")
        || mnemonic.starts_with("stp")
        || mnemonic.starts_with("stur")
        || mnemonic.starts_with("stlr")
        || mnemonic.starts_with("st1")
    {
        return "store";
    }
    "neutral"
}

fn assert_disasm_follows_rule(disasm: &str) {
    let mut dirty = true;
    for line in disasm.lines() {
        let line = line.trim();
        if line.is_empty() || line.ends_with(':') {
            continue;
        }
        let mnemonic = line.split_whitespace().next().unwrap_or("");
        match step_kind(mnemonic) {
            "barrier" => dirty = false,
            "load" => {
                assert!(!dirty, "load without SSBB:\n{line}\n{disasm}");
                dirty = false;
            }
            "store" => dirty = true,
            "atomic" | "call" => {
                assert!(!dirty, "atomic or call without SSBB:\n{line}\n{disasm}");
                dirty = true;
            }
            _ => {}
        }
    }
}

#[derive(Debug)]
struct Compiled {
    bytes: Vec<u8>,
    disasm: String,
    traps: usize,
}

fn target(mode: &str, has_lse: bool, frame_pointers: bool) -> crate::isa::OwnedTargetIsa {
    let mut shared = settings::builder();
    shared.set("opt_level", "none").unwrap();
    if frame_pointers {
        shared.set("preserve_frame_pointers", "true").unwrap();
    }
    let triple = "aarch64".parse().unwrap();
    let mut builder = isa::lookup(triple).unwrap();
    builder
        .set("has_lse", if has_lse { "true" } else { "false" })
        .unwrap();
    builder.set("aarch64_mitigation", mode).unwrap();
    builder.finish(settings::Flags::new(shared)).unwrap()
}

fn compile_func(
    mode: &str,
    has_lse: bool,
    frame_pointers: bool,
    func: Function,
) -> Result<Compiled, CodegenError> {
    let isa = target(mode, has_lse, frame_pointers);
    let mut ctx = Context::for_function(func);
    ctx.set_disasm(true);
    match ctx.compile(&*isa, &mut ControlPlane::default()) {
        Ok(code) => Ok(Compiled {
            bytes: code.buffer.data().to_vec(),
            disasm: code.vcode.clone().unwrap_or_default(),
            traps: code.buffer.traps().len(),
        }),
        Err(err) => Err(err.inner),
    }
}

fn must_compile(mode: &str, has_lse: bool, frame_pointers: bool, func: Function) -> Compiled {
    match compile_func(mode, has_lse, frame_pointers, func) {
        Ok(code) => code,
        Err(err) => panic!("compile failed: {err:?}"),
    }
}

fn new_func(params: &[Type], returns: &[Type]) -> Function {
    let mut sig = Signature::new(CallConv::SystemV);
    for ty in params {
        sig.params.push(AbiParam::new(*ty));
    }
    for ty in returns {
        sig.returns.push(AbiParam::new(*ty));
    }
    Function::with_name_signature(UserFuncName::user(0, 0), sig)
}

fn entry<'a>(func: &'a mut Function, params: &[Type]) -> (FuncCursor<'a>, Vec<Value>) {
    let block = func.dfg.make_block();
    let mut values = Vec::new();
    for ty in params {
        values.push(func.dfg.append_block_param(block, *ty));
    }
    let mut pos = FuncCursor::new(func);
    pos.insert_block(block);
    (pos, values)
}

fn no_args() -> &'static [BlockArg] {
    &[]
}

fn import_func(func: &mut Function, sig: Signature) -> FuncRef {
    let signature = func.import_signature(sig);
    let name = func.declare_imported_user_function(UserExternalName::new(0, 0));
    func.import_function(ExtFuncData {
        name: ExternalName::user(name),
        signature,
        colocated: true,
        patchable: false,
    })
}

fn store_store_load_load() -> Function {
    let mut func = new_func(&[types::I64, types::I64], &[types::I64]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
        let ptr = vals[0];
        let val = vals[1];
        pos.ins().store(MemFlags::trusted(), val, ptr, 0);
        pos.ins().store(MemFlags::trusted(), val, ptr, 8);
        let a = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
        let b = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 8);
        let sum = pos.ins().iadd(a, b);
        pos.ins().return_(&[sum]);
    }
    func
}

#[test]
fn none_adds_no_experiment_barrier() {
    let code = must_compile("none", true, false, store_store_load_load());
    assert_eq!(count_word(&code.bytes, SSBB), 0, "{}", code.disasm);
    assert_eq!(count_word(&code.bytes, CSDB), 0, "{}", code.disasm);
}

#[test]
fn cas_csdb_skips_function_without_native_cas() {
    let code = must_compile("cas_csdb", true, false, store_store_load_load());
    assert_eq!(count_word(&code.bytes, CSDB), 0, "{}", code.disasm);
    assert_eq!(count_word(&code.bytes, SSBB), 0, "{}", code.disasm);
}

#[test]
fn cas_csdb_follows_every_width() {
    for ty in [types::I8, types::I16, types::I32, types::I64] {
        let mut func = new_func(&[types::I64, ty, ty], &[ty]);
        {
            let (mut pos, vals) = entry(&mut func, &[types::I64, ty, ty]);
            let old = pos
                .ins()
                .atomic_cas(MemFlags::trusted(), vals[0], vals[1], vals[2]);
            pos.ins().return_(&[old]);
        }
        let code = must_compile("cas_csdb", true, false, func);
        let insns = words(&code.bytes);
        let mut cas = 0;
        for pair in insns.windows(2) {
            if is_cas(pair[0]) {
                assert_eq!(
                    pair[1], CSDB,
                    "CAS {ty} not followed by CSDB\n{}",
                    code.disasm
                );
                cas += 1;
            }
        }
        assert!(cas >= 1, "no native CAS for {ty}\n{}", code.disasm);
        assert_eq!(count_word(&code.bytes, SSBB), 0);
    }
}

#[test]
fn ssbb_store_pairs_and_alternation() {
    let coalesced = must_compile("ssbb", true, false, store_store_load_load());
    assert_eq!(
        count_word(&coalesced.bytes, SSBB),
        1,
        "{}",
        coalesced.disasm
    );
    assert_disasm_follows_rule(&coalesced.disasm);

    let mut func = new_func(&[types::I64, types::I64], &[types::I64]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
        pos.ins().store(MemFlags::trusted(), vals[1], vals[0], 0);
        let a = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 0);
        pos.ins().store(MemFlags::trusted(), a, vals[0], 8);
        let b = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 8);
        pos.ins().return_(&[b]);
    }
    let alternated = must_compile("ssbb", true, false, func);
    assert_eq!(
        count_word(&alternated.bytes, SSBB),
        2,
        "{}",
        alternated.disasm
    );
    assert_disasm_follows_rule(&alternated.disasm);
}

#[test]
fn ssbb_entry_loads_include_frame_ops() {
    let mut func = new_func(&[types::I64], &[types::I64]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64]);
        let a = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 0);
        let b = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 8);
        let sum = pos.ins().iadd(a, b);
        pos.ins().return_(&[sum]);
    }
    let code = must_compile("ssbb", true, true, func);
    assert_eq!(count_word(&code.bytes, SSBB), 1, "{}", code.disasm);
    assert!(code.disasm.contains("stp "), "{}", code.disasm);
    assert!(code.disasm.contains("ldp "), "{}", code.disasm);
    assert_disasm_follows_rule(&code.disasm);

    let mut func = new_func(&[types::I64, types::I64], &[]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
        pos.ins().store(MemFlags::trusted(), vals[1], vals[0], 0);
        pos.ins().return_(&[]);
    }
    let code = must_compile("ssbb", true, true, func);
    assert_eq!(count_word(&code.bytes, SSBB), 1, "{}", code.disasm);
    assert_disasm_follows_rule(&code.disasm);
}

#[test]
fn ssbb_join_and_loop_back_edge() {
    let mut func = new_func(&[types::I64, types::I32], &[types::I64]);
    let b0 = func.dfg.make_block();
    let ptr = func.dfg.append_block_param(b0, types::I64);
    let flag = func.dfg.append_block_param(b0, types::I32);
    let b_store = func.dfg.make_block();
    let b_clear = func.dfg.make_block();
    let b_join = func.dfg.make_block();
    {
        let mut pos = FuncCursor::new(&mut func);
        pos.insert_block(b0);
        let cleared = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
        let cond = pos.ins().icmp_imm(IntCC::NotEqual, flag, 0);
        pos.ins().brif(cond, b_store, no_args(), b_clear, no_args());
        pos.insert_block(b_store);
        pos.ins().store(MemFlags::trusted(), cleared, ptr, 0);
        pos.ins().jump(b_join, no_args());
        pos.insert_block(b_clear);
        pos.ins().jump(b_join, no_args());
        pos.insert_block(b_join);
        let joined = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 8);
        pos.ins().return_(&[joined]);
    }
    for policy in ["frame", "ranges", "ranges_cfg"] {
        let joined = compile_policy(policy, func.clone());
        assert_eq!(
            count_word(&joined.bytes, SSBB),
            2,
            "{policy}: {}",
            joined.disasm
        );
    }
    let join = must_compile("ssbb", true, false, func);
    assert_eq!(count_word(&join.bytes, SSBB), 2, "{}", join.disasm);
    assert_disasm_follows_rule(&join.disasm);

    let mut func = new_func(&[types::I64, types::I32], &[types::I64]);
    let b0 = func.dfg.make_block();
    let ptr = func.dfg.append_block_param(b0, types::I64);
    let flag = func.dfg.append_block_param(b0, types::I32);
    let b_loop = func.dfg.make_block();
    let b_exit = func.dfg.make_block();
    {
        let mut pos = FuncCursor::new(&mut func);
        pos.insert_block(b0);
        let cleared = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
        let cond = pos.ins().icmp_imm(IntCC::Equal, flag, 0);
        pos.ins().brif(cond, b_exit, no_args(), b_loop, no_args());
        pos.insert_block(b_loop);
        let next = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 8);
        pos.ins().store(MemFlags::trusted(), next, ptr, 0);
        pos.ins().jump(b_loop, no_args());
        pos.insert_block(b_exit);
        pos.ins().return_(&[cleared]);
    }
    for policy in ["frame", "ranges", "ranges_cfg"] {
        let looped = compile_policy(policy, func.clone());
        assert_eq!(
            count_word(&looped.bytes, SSBB),
            2,
            "{policy} must retain protection on the store backedge: {}",
            looped.disasm
        );
    }
    let looped = must_compile("ssbb", true, false, func);
    assert_eq!(count_word(&looped.bytes, SSBB), 2, "{}", looped.disasm);
    assert_disasm_follows_rule(&looped.disasm);
}

#[test]
fn ssbb_call_atomic_and_existing_dmb() {
    let mut func = new_func(&[types::I64, types::I64], &[types::I64]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
        let callee = import_func(pos.func, Signature::new(CallConv::SystemV));
        pos.ins().store(MemFlags::trusted(), vals[1], vals[0], 0);
        pos.ins().call(callee, &[]);
        let loaded = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 0);
        pos.ins().return_(&[loaded]);
    }
    let code = must_compile("ssbb", true, false, func);
    let call_at = code.disasm.find("bl ").expect(&code.disasm);
    assert!(code.disasm[..call_at].contains("dsb #0"), "{}", code.disasm);
    assert!(code.disasm[call_at..].contains("dsb #0"), "{}", code.disasm);
    assert_disasm_follows_rule(&code.disasm);

    let mut func = new_func(&[types::I64, types::I64], &[types::I64]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
        let updated = pos.ins().atomic_rmw(
            types::I64,
            MemFlags::trusted(),
            AtomicRmwOp::Add,
            vals[0],
            vals[1],
        );
        let loaded = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 8);
        let sum = pos.ins().iadd(updated, loaded);
        pos.ins().return_(&[sum]);
    }
    let code = must_compile("ssbb", true, false, func);
    assert_eq!(count_word(&code.bytes, SSBB), 2, "{}", code.disasm);
    assert_eq!(count_word(&code.bytes, CSDB), 0, "{}", code.disasm);
    assert_disasm_follows_rule(&code.disasm);

    for ty in [types::I32, types::I64] {
        let mut func = new_func(&[types::I64, ty], &[ty]);
        {
            let (mut pos, vals) = entry(&mut func, &[types::I64, ty]);
            let updated =
                pos.ins()
                    .atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Xchg, vals[0], vals[1]);
            pos.ins().return_(&[updated]);
        }
        let code = must_compile("ssbb", true, false, func);
        assert!(
            code.disasm.contains("swpal"),
            "LSE exchange should be swpal, got {}",
            code.disasm
        );
        assert!(
            !code.disasm.contains("ldaxr") && !code.disasm.contains("stlxr"),
            "exchange must not stay on the LL/SC loop: {}",
            code.disasm
        );
        assert_disasm_follows_rule(&code.disasm);
    }

    let mut func = new_func(&[types::I64, types::I64], &[types::I64]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
        pos.ins().store(MemFlags::trusted(), vals[1], vals[0], 0);
        pos.ins().fence();
        let loaded = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 0);
        pos.ins().return_(&[loaded]);
    }
    let code = must_compile("ssbb", true, false, func);
    assert_eq!(count_word(&code.bytes, SSBB), 1, "{}", code.disasm);
    assert!(code.disasm.contains("dmb ish"), "{}", code.disasm);
    assert_disasm_follows_rule(&code.disasm);
}

#[test]
fn ssbb_spills_vectors_and_multiple_returns() {
    let mut func = new_func(&[types::I64], &[types::I64]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64]);
        let callee = import_func(pos.func, Signature::new(CallConv::SystemV));
        let mut live = Vec::new();
        for i in 0..20 {
            live.push(
                pos.ins()
                    .load(types::I64, MemFlags::trusted(), vals[0], i * 8),
            );
        }
        pos.ins().call(callee, &[]);
        let mut acc = live[0];
        for value in live.into_iter().skip(1) {
            acc = pos.ins().iadd(acc, value);
        }
        pos.ins().return_(&[acc]);
    }
    for policy in ["frame", "ranges", "ranges_cfg"] {
        let spilled = compile_policy(policy, func.clone());
        assert!(
            count_word(&spilled.bytes, SSBB) >= 2,
            "{policy} must protect the call and regalloc reloads: {}",
            spilled.disasm
        );
        assert!(
            spilled.disasm.contains("str ") || spilled.disasm.contains("stur "),
            "{}",
            spilled.disasm
        );
    }
    let spilled = must_compile("ssbb", true, false, func);
    assert!(
        spilled.disasm.contains("str ") || spilled.disasm.contains("stur "),
        "{}",
        spilled.disasm
    );
    assert_disasm_follows_rule(&spilled.disasm);

    let mut func = new_func(
        &[types::I64, types::I8X16, types::F64, types::I8],
        &[types::I8],
    );
    {
        let (mut pos, vals) = entry(
            &mut func,
            &[types::I64, types::I8X16, types::F64, types::I8],
        );
        pos.ins().store(MemFlags::trusted(), vals[1], vals[0], 0);
        let vec = pos
            .ins()
            .load(types::I8X16, MemFlags::trusted(), vals[0], 0);
        pos.ins().store(MemFlags::trusted(), vec, vals[0], 16);
        pos.ins().store(MemFlags::trusted(), vals[2], vals[0], 32);
        let fp = pos.ins().load(types::F64, MemFlags::trusted(), vals[0], 32);
        pos.ins().store(MemFlags::trusted(), fp, vals[0], 40);
        pos.ins().store(MemFlags::trusted(), vals[3], vals[0], 48);
        let narrow = pos.ins().load(types::I8, MemFlags::trusted(), vals[0], 48);
        pos.ins().return_(&[narrow]);
    }
    let wide = must_compile("ssbb", true, true, func);
    assert_disasm_follows_rule(&wide.disasm);
    assert!(
        wide.disasm.contains("stp ") || wide.disasm.contains("ldp "),
        "{}",
        wide.disasm
    );

    let mut func = new_func(&[types::I32], &[]);
    let b0 = func.dfg.make_block();
    let flag = func.dfg.append_block_param(b0, types::I32);
    let b1 = func.dfg.make_block();
    let b2 = func.dfg.make_block();
    {
        let mut pos = FuncCursor::new(&mut func);
        pos.insert_block(b0);
        let cond = pos.ins().icmp_imm(IntCC::Equal, flag, 0);
        pos.ins().brif(cond, b1, no_args(), b2, no_args());
        pos.insert_block(b1);
        pos.ins().return_(&[]);
        pos.insert_block(b2);
        pos.ins().return_(&[]);
    }
    let multi = must_compile("ssbb", true, true, func);
    assert_disasm_follows_rule(&multi.disasm);
    assert!(count_word(&multi.bytes, SSBB) >= 1, "{}", multi.disasm);
}

#[test]
fn ssbb_jump_table_calls_traps_and_veneer() {
    let mut func = new_func(&[types::I32], &[types::I32]);
    let b0 = func.dfg.make_block();
    let idx = func.dfg.append_block_param(b0, types::I32);
    let arms: Vec<_> = (0..4).map(|_| func.dfg.make_block()).collect();
    let exit = func.dfg.make_block();
    let value = func.dfg.append_block_param(exit, types::I32);
    let jt = {
        let default = func.dfg.block_call(arms[0], no_args());
        let table: Vec<_> = arms
            .iter()
            .map(|block| func.dfg.block_call(*block, no_args()))
            .collect();
        func.create_jump_table(JumpTableData::new(default, &table))
    };
    {
        let mut pos = FuncCursor::new(&mut func);
        pos.insert_block(b0);
        pos.ins().br_table(idx, jt);
        for (i, block) in arms.iter().enumerate() {
            pos.insert_block(*block);
            let imm = pos.ins().iconst(types::I32, i as i64);
            let arg = [BlockArg::Value(imm)];
            pos.ins().jump(exit, &arg);
        }
        pos.insert_block(exit);
        pos.ins().return_(&[value]);
    }
    let table = must_compile("ssbb", true, false, func);
    let insns = words(&table.bytes);
    let ssbb_at = insns
        .iter()
        .position(|word| *word == SSBB)
        .expect(&table.disasm);
    let csdb_at = insns
        .iter()
        .position(|word| *word == CSDB)
        .expect(&table.disasm);
    let adr_at = insns
        .iter()
        .position(|word| is_adr16(*word))
        .expect(&table.disasm);
    assert!(ssbb_at < csdb_at && csdb_at < adr_at, "{}", table.disasm);
    assert_disasm_follows_rule(&table.disasm);

    let mut func = new_func(&[types::I64, types::I64], &[types::I64]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
        let sig = pos.func.import_signature(Signature::new(CallConv::SystemV));
        pos.ins().store(MemFlags::trusted(), vals[1], vals[0], 0);
        pos.ins().call_indirect(sig, vals[1], &[]);
        let loaded = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 8);
        pos.ins().return_(&[loaded]);
    }
    let indirect = must_compile("ssbb", true, false, func);
    assert!(indirect.disasm.contains("blr "), "{}", indirect.disasm);
    assert_disasm_follows_rule(&indirect.disasm);

    let mut func = new_func(&[types::I64, types::I64], &[]);
    func.signature.call_conv = CallConv::Tail;
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64]);
        let callee = import_func(pos.func, Signature::new(CallConv::Tail));
        pos.ins().store(MemFlags::trusted(), vals[1], vals[0], 0);
        pos.ins().return_call(callee, &[]);
    }
    let tail = must_compile("ssbb", true, false, func);
    assert!(tail.disasm.contains("return_call"), "{}", tail.disasm);
    assert_disasm_follows_rule(&tail.disasm);

    let mut func = new_func(&[types::I64], &[]);
    {
        let (mut pos, vals) = entry(&mut func, &[types::I64]);
        pos.ins().store(MemFlags::trusted(), vals[0], vals[0], 0);
        pos.ins().trap(TrapCode::HEAP_OUT_OF_BOUNDS);
    }
    let trapped = must_compile("ssbb", true, false, func);
    assert!(trapped.traps >= 1, "{}", trapped.disasm);
    assert_disasm_follows_rule(&trapped.disasm);

    let mut buf = MachBuffer::<Inst>::new();
    buf.set_veneer_load_barrier(Some(SSBB));
    buf.put4(0);
    let label = buf.get_label();
    buf.testing_emit_veneer(label, 0, LabelUse::Branch26);
    let data = buf.testing_data();
    assert_eq!(&data[4..8], &SSBB.to_le_bytes());
    let ldr = enc_ldst_imm19(0b1001_1000, 4, spilltmp_reg());
    assert_eq!(u32::from_le_bytes(data[8..12].try_into().unwrap()), ldr);
    assert_eq!(data.len(), 4 + 24);
}

#[test]
fn rejected_modes_fail_explicitly() {
    let mut builder = crate::isa::aarch64::settings::builder();
    assert!(builder.set("aarch64_mitigation", "bogus").is_err());

    let err = compile_func("ssbb", false, false, store_store_load_load()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("has_lse"), "{text}");

    let err = compile_func("cas_csdb", false, false, store_store_load_load()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("has_lse"), "{text}");

    let err = classify(
        &Inst::AtomicCASLoop {
            ty: types::I64,
            flags: MemFlags::trusted(),
            addr: xreg(25),
            expected: xreg(26),
            replacement: xreg(28),
            oldval: writable_xreg(27),
            scratch: writable_xreg(24),
        },
        &FrameLayout::default(),
        &settings::Flags::new(settings::builder()),
    )
    .unwrap_err();
    assert!(err.to_string().contains("LL/SC"), "{err}");

    let flags = settings::Flags::new(settings::builder());
    let frame = FrameLayout::default();
    assert!(
        classify(&Inst::Csdb {}, &frame, &flags)
            .unwrap()
            .steps
            .is_empty()
    );
    assert!(
        classify(&Inst::Fence {}, &frame, &flags)
            .unwrap()
            .steps
            .is_empty()
    );
    assert_eq!(
        classify(&Inst::Ssbb {}, &frame, &flags)
            .unwrap()
            .steps
            .as_slice(),
        &[SsbbMemStep::Clear]
    );
    assert_eq!(
        SsbbSeq::count_barriers(
            true,
            &[SsbbMemStep::Store, SsbbMemStep::Clear, SsbbMemStep::Load]
        ),
        0
    );
    assert_eq!(
        SsbbSeq::count_barriers(true, &[SsbbMemStep::Store, SsbbMemStep::Load]),
        1
    );

    // This Cranelift version rejects SystemV returns that do not fit in
    // registers, so stack-return calls are checked on the shared atom list
    // the emitter walks when a call is internal.
    let info = crate::machinst::abi::CallInfo {
        dest: xreg(0),
        uses: smallvec::smallvec![],
        defs: smallvec::smallvec![crate::machinst::abi::CallRetPair {
            vreg: writable_xreg(1),
            location: crate::machinst::abi::RetLocation::Stack(
                crate::machinst::abi::StackAMode::Slot(0),
                types::I64,
            ),
        }],
        clobbers: Default::default(),
        callee_conv: CallConv::SystemV,
        caller_conv: CallConv::SystemV,
        callee_pop_size: 0,
        try_call_info: None,
        patchable: false,
    };
    let ops = call_retval_ops(&info, 0);
    assert!(ops.iter().any(|op| matches!(
        op,
        RetvalOp::Mem {
            step: SsbbMemStep::Load,
            ..
        }
    )));
    let mut seq = SsbbSeq::neutral();
    seq.steps.push(SsbbMemStep::Call);
    for op in ops {
        if let RetvalOp::Mem { step, .. } = op {
            seq.steps.push(step);
        }
    }
    assert!(seq.needs_internal());
    assert_eq!(
        SsbbSeq::count_barriers(true, &seq.steps),
        2,
        "SSBB before the call and again before the stack return load"
    );
}

fn target_opt(mode: &str, opt: &str) -> crate::isa::OwnedTargetIsa {
    let mut shared = settings::builder();
    shared.set("opt_level", opt).unwrap();
    let triple = "aarch64".parse().unwrap();
    let mut builder = isa::lookup(triple).unwrap();
    builder.set("has_lse", "true").unwrap();
    builder.set("aarch64_mitigation", mode).unwrap();
    builder.finish(settings::Flags::new(shared)).unwrap()
}

fn compile_opt(mode: &str, opt: &str, func: Function) -> Compiled {
    let isa = target_opt(mode, opt);
    let mut ctx = Context::for_function(func);
    ctx.set_disasm(true);
    let code = match ctx.compile(&*isa, &mut ControlPlane::default()) {
        Ok(code) => code,
        Err(err) => panic!("compile {mode}/{opt} failed: {:?}", err.inner),
    };
    let relocs = code.buffer.relocs().len();
    assert!(
        relocs == 0,
        "{mode}/{opt} has {relocs} relocs; blob is not a leaf:\n{}",
        code.vcode.clone().unwrap_or_default()
    );
    Compiled {
        bytes: code.buffer.data().to_vec(),
        disasm: code.vcode.clone().unwrap_or_default(),
        traps: code.buffer.traps().len(),
    }
}

fn gap64() -> Function {
    let mut func = new_func(&[types::I64, types::I64, types::I64], &[types::I64]);
    let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64, types::I64]);
    let old = pos
        .ins()
        .atomic_cas(MemFlags::trusted(), vals[0], vals[1], vals[2]);
    let ptr = pos.ins().load(types::I64, MemFlags::trusted(), vals[0], 0);
    let touched = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
    let keep = pos.ins().iadd(old, touched);
    pos.ins().return_(&[keep]);
    func
}

fn gap32() -> Function {
    let mut func = new_func(&[types::I64, types::I32, types::I32], &[types::I32]);
    let (mut pos, vals) = entry(&mut func, &[types::I64, types::I32, types::I32]);
    let old = pos
        .ins()
        .atomic_cas(MemFlags::trusted(), vals[0], vals[1], vals[2]);
    pos.ins().return_(&[old]);
    func
}

fn v4_gadget() -> Function {
    let mut func = new_func(
        &[types::I64, types::I64, types::I64, types::I64],
        &[types::I64],
    );
    let (mut pos, vals) = entry(&mut func, &[types::I64, types::I64, types::I64, types::I64]);
    let mut addr = vals[0];
    for _ in 0..40 {
        addr = pos.ins().imul(addr, vals[1]);
    }
    pos.ins().store(MemFlags::trusted(), vals[2], addr, 0);
    let ptr = pos.ins().load(types::I64, MemFlags::trusted(), vals[3], 0);
    let touched = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
    pos.ins().return_(&[touched]);
    func
}

fn mnemonic_count(disasm: &str, mnemonic: &str) -> usize {
    disasm
        .lines()
        .filter(|line| line.split_whitespace().next() == Some(mnemonic))
        .count()
}

fn delay_count(disasm: &str) -> usize {
    // Cranelift lowers `x * y` to `madd x, x, y, xzr`, which is still a
    // data-dependent multiply chain.
    mnemonic_count(disasm, "mul") + mnemonic_count(disasm, "madd")
}

fn assert_cas_shape(code: &Compiled, want_csdb: bool) {
    let insns = words(&code.bytes);
    let mut cas = 0;
    for (index, word) in insns.iter().enumerate() {
        if is_cas(*word) {
            cas += 1;
            let next = insns.get(index + 1).copied();
            if want_csdb {
                assert_eq!(
                    next,
                    Some(CSDB),
                    "CAS not followed by CSDB\n{}",
                    code.disasm
                );
            }
        }
    }
    assert!(cas >= 1, "no native CAS\n{}", code.disasm);
    assert_eq!(count_word(&code.bytes, SSBB), 0, "{}", code.disasm);
    if !want_csdb {
        assert_eq!(count_word(&code.bytes, CSDB), 0, "{}", code.disasm);
    }
}

fn emit_symbol(out: &mut String, name: &str, bytes: &[u8]) {
    assert!(bytes.len() % 4 == 0, "{name} length {}", bytes.len());
    out.push_str(&format!(".p2align 2\n.globl _{name}\n_{name}:\n"));
    for chunk in bytes.chunks_exact(4) {
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        out.push_str(&format!(".inst 0x{word:08x}\n"));
    }
    out.push('\n');
}

fn write_disasm(dir: &str, name: &str, opt: &str, code: &Compiled) {
    let path = format!("{dir}/{name}.disasm");
    let text = format!(
        "name={name}\nopt={opt}\nbytes={}\ntraps={}\n\n{}",
        code.bytes.len(),
        code.traps,
        code.disasm
    );
    std::fs::write(&path, text).unwrap();
}

/// Dump leaf gadgets when GADGET_DUMP_DIR is set. No-op in a normal test run.
#[test]
fn gadget_phase_e_dump() {
    let Ok(dir) = std::env::var("GADGET_DUMP_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();

    let gap_none = compile_opt("none", "speed", gap64());
    let gap_cas = compile_opt("cas_csdb", "speed", gap64());
    assert_cas_shape(&gap_none, false);
    assert_cas_shape(&gap_cas, true);
    write_disasm(&dir, "gap64-none-speed", "speed", &gap_none);
    write_disasm(&dir, "gap64-cas-speed", "speed", &gap_cas);

    let gap32_cas = compile_opt("cas_csdb", "speed", gap32());
    assert_cas_shape(&gap32_cas, true);
    write_disasm(&dir, "gap32-cas-speed", "speed", &gap32_cas);

    let v4_speed_none = compile_opt("none", "speed", v4_gadget());
    let v4_speed_ssbb = compile_opt("ssbb", "speed", v4_gadget());
    let v4_none_none = compile_opt("none", "none", v4_gadget());
    let v4_none_ssbb = compile_opt("ssbb", "none", v4_gadget());
    for (label, code) in [
        ("v4-none-speed", &v4_speed_none),
        ("v4-ssbb-speed", &v4_speed_ssbb),
        ("v4-none-optnone", &v4_none_none),
        ("v4-ssbb-optnone", &v4_none_ssbb),
    ] {
        let opt = if label.ends_with("speed") {
            "speed"
        } else {
            "none"
        };
        write_disasm(&dir, label, opt, code);
    }

    let speed_muls = delay_count(&v4_speed_none.disasm);
    let none_muls = delay_count(&v4_none_none.disasm);
    let (v4_opt, v4_none_code, v4_ssbb_code) = if speed_muls >= 20 {
        ("speed", &v4_speed_none, &v4_speed_ssbb)
    } else {
        assert!(
            none_muls >= 20,
            "neither opt kept a mul delay: speed={speed_muls} none={none_muls}\n{}",
            v4_none_none.disasm
        );
        ("none", &v4_none_none, &v4_none_ssbb)
    };
    assert_eq!(
        count_word(&v4_none_code.bytes, SSBB),
        0,
        "{}",
        v4_none_code.disasm
    );
    assert!(
        count_word(&v4_ssbb_code.bytes, SSBB) >= 1,
        "ssbb gadget has no SSBB\n{}",
        v4_ssbb_code.disasm
    );
    assert_eq!(count_word(&v4_none_code.bytes, CSDB), 0);
    assert_eq!(count_word(&v4_ssbb_code.bytes, CSDB), 0);

    let mut asm = String::from(".section __TEXT,__text,regular,pure_instructions\n\n");
    emit_symbol(&mut asm, "gap_backend_none", &gap_none.bytes);
    emit_symbol(&mut asm, "gap_backend_cas", &gap_cas.bytes);
    emit_symbol(&mut asm, "v4_backend_none", &v4_none_code.bytes);
    emit_symbol(&mut asm, "v4_backend_ssbb", &v4_ssbb_code.bytes);
    std::fs::write(format!("{dir}/blobs.S"), asm).unwrap();
    std::fs::write(
        format!("{dir}/chosen.txt"),
        format!(
            "gap_opt=speed\nv4_opt={v4_opt}\nv4_mul_speed={speed_muls}\nv4_mul_none={none_muls}\ngap_none_bytes={}\ngap_cas_bytes={}\nv4_none_bytes={}\nv4_ssbb_bytes={}\n",
            gap_none.bytes.len(),
            gap_cas.bytes.len(),
            v4_none_code.bytes.len(),
            v4_ssbb_code.bytes.len()
        ),
    )
    .unwrap();
}
