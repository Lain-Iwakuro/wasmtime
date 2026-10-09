//! End-to-end regressions for the experimental store-to-load LFENCE policy.
use crate::Context;
use crate::cursor::{Cursor, FuncCursor};
use crate::ir::condcodes::IntCC;
use crate::ir::{
    AbiParam, BlockArg, ExtFuncData, ExternalName, FuncRef, Function, InstBuilder, MemFlags,
    Signature, Type, UserExternalName, UserFuncName, Value, types,
};
use crate::isa::{self, CallConv};
use crate::settings::{self, Configurable};
use alloc::boxed::Box;
use alloc::vec::Vec;
use cranelift_control::ControlPlane;

fn compile(func: Function, mode: Option<&str>) -> Vec<u8> {
    compile_policy(func, mode, "legacy")
}

fn compile_policy(func: Function, mode: Option<&str>, policy: &str) -> Vec<u8> {
    let mut shared = settings::builder();
    shared.set("opt_level", "none").unwrap();
    shared.set("v4_fence_policy", policy).unwrap();
    let mut builder = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap()).unwrap();
    if let Some(mode) = mode {
        builder.set("x64_mitigation", mode).unwrap();
    }
    let isa = builder.finish(settings::Flags::new(shared)).unwrap();
    let mut ctx = Context::for_function(func);
    ctx.compile(&*isa, &mut ControlPlane::default())
        .unwrap()
        .buffer
        .data()
        .to_vec()
}

#[test]
fn selective_native_metadata_keeps_narrow_width_and_old_ea_epoch() {
    use super::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy, V4Range};
    let info = raw_info();
    let mut context = V4Context::new(3, V4Policy::Ranges);
    let memory = || Amode::imm_reg(16, regs::rdi());
    let load = Inst::External {
        inst: asm::inst::movzbq_rm::new(Writable::from_reg(Gpr::RDI), memory()).into(),
    };
    let seq =
        super::ssbb::classify_v4(&load, &FrameLayout::default(), &info, &mut context).unwrap();
    assert_eq!(
        seq.events[0],
        V4Event::Load(V4Range::LocalBase {
            block: 3,
            reg: regs::rdi(),
            epoch: 0,
            offset: 16,
            width: 1,
        })
    );
    assert_eq!(seq.events[1], V4Event::RegisterDef(regs::rdi()));
    let partial = Inst::External {
        inst: asm::inst::movb_mr::new(Writable::from_reg(Gpr::RDI), Gpr::RSI).into(),
    };
    super::ssbb::classify_v4(&partial, &FrameLayout::default(), &info, &mut context).unwrap();
    assert_eq!(
        context.local_base(regs::rdi(), 0, 8),
        V4Range::LocalBase {
            block: 3,
            reg: regs::rdi(),
            epoch: 2,
            offset: 0,
            width: 8,
        }
    );
}

#[test]
fn selective_native_frame_tracks_push_pop_ret_without_recovering_reloaded_fp() {
    use super::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy, V4Range};
    let info = raw_info();
    let mut context = V4Context::new(0, V4Policy::Frame);
    context.external_entry = true;
    let instructions = [
        asm::inst::pushq_o::new(Gpr::RBP).into(),
        asm::inst::movq_mr::new(Writable::from_reg(Gpr::RBP), Gpr::RSP).into(),
        asm::inst::popq_o::new(Writable::from_reg(Gpr::RBP)).into(),
        asm::inst::retq_zo::new().into(),
    ];
    let mut events = Vec::new();
    for inst in instructions {
        let seq = super::ssbb::classify_v4(
            &Inst::External { inst },
            &FrameLayout::default(),
            &info,
            &mut context,
        )
        .unwrap();
        events.extend(seq.events);
    }
    assert!(events.contains(&V4Event::Store(V4Range::Frame {
        frame: 0,
        offset: -8,
        width: 8
    })));
    assert!(events.contains(&V4Event::Load(V4Range::Frame {
        frame: 0,
        offset: -8,
        width: 8
    })));
    assert!(events.contains(&V4Event::Load(V4Range::Frame {
        frame: 0,
        offset: 0,
        width: 8
    })));
    assert_eq!(context.frame_offset(regs::rbp()), None);
}

#[test]
fn selective_frame_resolves_emission_known_offsets_and_rejects_indexed_addresses() {
    use super::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy, V4Range};
    let info = raw_info();
    let frame = FrameLayout {
        outgoing_args_size: 24,
        tail_args_size: 16,
        setup_area_size: 16,
        ..FrameLayout::default()
    };
    let mut context = V4Context::new(1, V4Policy::Ranges);
    context.set_frame_offset(regs::rsp(), Some(-40));
    context.set_frame_offset(regs::rbp(), Some(-8));
    for (address, expected) in [
        (
            SyntheticAmode::SlotOffset { simm32: 8 },
            V4Range::Frame {
                frame: 0,
                offset: -8,
                width: 1,
            },
        ),
        (
            SyntheticAmode::IncomingArg { offset: 8 },
            V4Range::Frame {
                frame: 0,
                offset: 16,
                width: 1,
            },
        ),
    ] {
        let inst = Inst::External {
            inst: asm::inst::movzbq_rm::new(
                Writable::from_reg(Gpr::RAX),
                asm::Amode::<Gpr>::from(address),
            )
            .into(),
        };
        let seq = super::ssbb::classify_v4(&inst, &frame, &info, &mut context).unwrap();
        assert_eq!(seq.events[0], V4Event::Load(expected));
    }
    for address in [
        asm::Amode::ImmReg {
            base: Gpr::RDI,
            simm32: asm::AmodeOffsetPlusKnownOffset {
                simm32: 0.into(),
                offset: Some(255),
            },
            trap: None,
        },
        asm::Amode::ImmRegRegShift {
            base: Gpr::RDI,
            index: asm::NonRspGpr::new(Gpr::RSI),
            scale: asm::Scale::One,
            simm32: 0.into(),
            trap: None,
        },
        asm::Amode::RipRelative {
            target: asm::DeferredTarget::None,
        },
    ] {
        let inst = Inst::External {
            inst: asm::inst::movq_rm::new(Writable::from_reg(Gpr::RAX), address).into(),
        };
        assert_eq!(
            super::ssbb::classify_v4(&inst, &frame, &info, &mut context)
                .unwrap()
                .events[0],
            V4Event::Load(V4Range::Unknown)
        );
    }
}

#[test]
fn selective_atomic_keeps_internal_retry_guards_and_suppresses_ordinary_auto_guards() {
    use super::*;
    use crate::machinst::v4_fences::{V4Context, V4Policy};
    let info = raw_info();
    let atomic = Inst::AtomicRmwSeq {
        ty: types::I8,
        op: AtomicRmwSeqOp::Nand,
        mem: SyntheticAmode::real(Amode::imm_reg(0, regs::rdi())),
        operand: Gpr::RSI,
        temp: Writable::from_reg(Gpr::RDX),
        dst_old: Writable::from_reg(Gpr::RAX),
    };
    let seq = super::ssbb::classify_v4(
        &atomic,
        &FrameLayout::default(),
        &info,
        &mut V4Context::new(0, V4Policy::Ranges),
    )
    .unwrap();
    assert!(seq.legacy_internal);
    assert_eq!(seq.fixed_barriers(), 1);
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.v4_set_selective(true);
    state.v4_set_legacy_internal(true);
    let mut sink = MachBuffer::<Inst>::new();
    atomic.emit(&mut sink, &info, &mut state);
    state.v4_set_legacy_internal(false);
    // The shared plan emits the post-RMW guard. The ordinary load must not
    // silently add a second copy through the legacy emitter.
    state.ssbb_emit_barrier(&mut sink, &info);
    Inst::External {
        inst: asm::inst::movzbq_rm::new(
            Writable::from_reg(Gpr::RAX),
            Amode::imm_reg(0, regs::rdi()),
        )
        .into(),
    }
    .emit(&mut sink, &info, &mut state);
    let bytes = sink
        .finish(&Default::default(), &mut ControlPlane::default())
        .data()
        .to_vec();
    assert_eq!(fences(&bytes), 2, "{bytes:02x?}");
    let cas = bytes
        .windows(3)
        .position(|b| b == [0xf0, 0x0f, 0xb0])
        .unwrap();
    assert_eq!(&bytes[cas - 3..cas], &[0x0f, 0xae, 0xe8]);
    let branch = bytes.windows(2).position(|b| b == [0x0f, 0x85]).unwrap();
    let displacement = i32::from_le_bytes(bytes[branch + 2..branch + 6].try_into().unwrap());
    assert!((branch as i32 + 6 + displacement) as usize <= cas - 3);
}

#[test]
fn selective_policies_compile_after_register_allocation() {
    for policy in ["frame", "ranges", "ranges_cfg"] {
        let bytes = compile_policy(store_load(), Some("lfence"), policy);
        assert!(fences(&bytes) > 0, "{policy}: {bytes:02x?}");
    }
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
    let cs = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap())
        .unwrap()
        .finish(settings::Flags::new(settings::builder()))
        .unwrap()
        .to_capstone()
        .unwrap();
    let reference = compile_policy(func.clone(), Some("lfence"), "ranges");
    let optimized = compile_policy(func, Some("lfence"), "ranges_cfg");
    assert_eq!(
        decoded_nonfence_stream(&cs, &optimized, &[0x0f, 0xae, 0xe8]),
        decoded_nonfence_stream(&cs, &reference, &[0x0f, 0xae, 0xe8]),
        "an empty retry edge must remain eligible for native branch folding"
    );
}

#[test]
fn selective_cfg_layout_separator_rejects_zero_code_and_complex_tails() {
    use super::*;
    let ordinary = Inst::External {
        inst: asm::inst::movq_mr::new(Amode::imm_reg(0, regs::rdi()), Gpr::RSI).into(),
    };
    assert!(ordinary.v4_emits_nonbranch_code());
    for inst in [
        Inst::Args { args: vec![] },
        Inst::DummyUse { reg: regs::rdi() },
        Inst::SequencePoint,
        Inst::External {
            inst: asm::inst::jmpq_m::new(Amode::imm_reg(0, regs::rdi())).into(),
        },
        Inst::External {
            inst: asm::inst::callq_m::new(Amode::imm_reg(0, regs::rdi())).into(),
        },
        Inst::External {
            inst: asm::inst::retq_zo::new().into(),
        },
        Inst::StackProbeLoop {
            tmp: Writable::from_reg(regs::rax()),
            frame_size: 8192,
            guard_size: 4096,
        },
        Inst::AtomicRmwSeq {
            ty: types::I8,
            op: AtomicRmwSeqOp::Nand,
            mem: SyntheticAmode::real(Amode::imm_reg(0, regs::rdi())),
            operand: Gpr::RSI,
            temp: Writable::from_reg(Gpr::RDX),
            dst_old: Writable::from_reg(Gpr::RAX),
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
    let cs = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap())
        .unwrap()
        .finish(settings::Flags::new(settings::builder()))
        .unwrap()
        .to_capstone()
        .unwrap();
    let reference = compile_policy(func.clone(), Some("lfence"), "ranges");
    let optimized = compile_policy(func, Some("lfence"), "ranges_cfg");
    assert_ne!(
        optimized, reference,
        "a store followed by a jump must allow CFG placement"
    );
    assert_eq!(
        decoded_nonfence_stream(&cs, &optimized, &[0x0f, 0xae, 0xe8]),
        decoded_nonfence_stream(&cs, &reference, &[0x0f, 0xae, 0xe8]),
    );
    let decoded = cs.disasm_all(&optimized, 0).unwrap();
    assert!(
        decoded.as_ref().windows(2).any(|pair| {
            pair[0].mnemonic() == Some("movq")
                && pair[0]
                    .op_str()
                    .unwrap()
                    .split_once(',')
                    .is_some_and(|(src, dst)| src.starts_with('%') && dst.contains('('))
                && pair[1].bytes() == [0x0f, 0xae, 0xe8]
        }),
        "the moved guard must follow the actual store"
    );
}

fn selective_native_plan(
    insts: &[super::Inst],
    policy: crate::machinst::v4_fences::V4Policy,
) -> (Vec<u8>, Vec<bool>) {
    selective_native_plan_with_frame(insts, policy, super::FrameLayout::default())
}

fn selective_native_plan_with_frame(
    insts: &[super::Inst],
    policy: crate::machinst::v4_fences::V4Policy,
    frame: super::FrameLayout,
) -> (Vec<u8>, Vec<bool>) {
    use super::*;
    use crate::machinst::v4_fences::{self, V4Block, V4Context, V4EventId, V4Graph, V4Node};
    let info = raw_info();
    let mut context = V4Context::new(0, policy);
    context.external_entry = true;
    let sequences: Vec<_> = insts
        .iter()
        .map(|inst| super::ssbb::classify_v4(inst, &frame, &info, &mut context).unwrap())
        .collect();
    let mut events = Vec::new();
    for (inst, seq) in sequences.iter().enumerate() {
        for (sub, event) in seq.events.iter().copied().enumerate() {
            events.push(V4Node {
                id: V4EventId {
                    block: 0,
                    inst: inst as u32,
                    sub: sub as u16,
                },
                event,
                // An expansion's first access is protected at its outer native
                // boundary; only its actual internal guards are frozen.
                guard: Some(inst),
            });
        }
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
    let plan = v4_fences::plan(&graph, policy).unwrap();
    assert!(v4_fences::check(&graph, &plan.fences).errors.is_empty());
    let mut state = EmitState::default();
    state.frame_layout = frame;
    state.ssbb_install(&[]);
    state.v4_set_selective(true);
    let mut sink = MachBuffer::<Inst>::new();
    for (site, (inst, seq)) in insts.iter().zip(&sequences).enumerate() {
        if plan.fences[site] {
            state.ssbb_emit_barrier(&mut sink, &info);
        }
        state.v4_set_legacy_internal(seq.legacy_internal);
        inst.emit(&mut sink, &info, &mut state);
        state.v4_set_legacy_internal(false);
    }
    (
        sink.finish(&Default::default(), &mut ControlPlane::default())
            .data()
            .to_vec(),
        plan.fences,
    )
}

fn selective_call_info<T>(dest: T) -> Box<super::CallInfo<T>> {
    use super::*;
    Box::new(CallInfo {
        dest,
        uses: smallvec![],
        defs: smallvec![],
        clobbers: Default::default(),
        callee_conv: CallConv::SystemV,
        caller_conv: CallConv::SystemV,
        callee_pop_size: 0,
        try_call_info: None,
        patchable: false,
    })
}

#[test]
fn selective_clean_load_and_pure_sequence_does_not_add_call_entry_fence() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    let insts = [
        Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        },
        Inst::External {
            inst: asm::inst::movq_rm::new(
                Writable::from_reg(Gpr::RAX),
                Amode::imm_reg(0, regs::rdi()),
            )
            .into(),
        },
        Inst::addq_mi(Writable::from_reg(regs::rax()), 7),
        Inst::CallKnown {
            info: selective_call_info(ExternalName::LibCall(LibCall::Memcpy)),
        },
    ];
    for policy in [V4Policy::Frame, V4Policy::Ranges, V4Policy::RangesCfg] {
        let (bytes, plan) = selective_native_plan(&insts, policy);
        assert_eq!(
            fences(&bytes),
            1,
            "clean call gained a redundant guard: {bytes:02x?}"
        );
        assert_eq!(plan, [false; 4]);
        assert_eq!(&bytes[bytes.len() - 5..], &[0xe8, 0, 0, 0, 0]);
        assert_eq!(&bytes[3..6], &[0x48, 0x8b, 0x07]);
    }
}

#[test]
fn selective_store_before_call_still_emits_call_entry_fence() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    let insts = [
        Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        },
        Inst::External {
            inst: asm::inst::movq_mr::new(Amode::imm_reg(0, regs::rdi()), Gpr::RSI).into(),
        },
        Inst::CallKnown {
            info: selective_call_info(ExternalName::LibCall(LibCall::Memcpy)),
        },
    ];
    let (bytes, _) = selective_native_plan(&insts, V4Policy::Ranges);
    assert_eq!(
        bytes,
        [
            0x0f, 0xae, 0xe8, 0x48, 0x89, 0x37, 0x0f, 0xae, 0xe8, 0xe8, 0, 0, 0, 0
        ]
    );
}

#[test]
fn selective_memory_call_target_is_read_and_guarded_at_native_call_boundary() {
    use super::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy};
    for dirty in [false, true] {
        let call = Inst::CallUnknown {
            info: selective_call_info(RegMem::Mem {
                addr: SyntheticAmode::real(Amode::imm_reg(0, regs::rdi())),
            }),
        };
        let seq = super::ssbb::classify_v4(
            &call,
            &FrameLayout::default(),
            &raw_info(),
            &mut V4Context::new(0, V4Policy::Ranges),
        )
        .unwrap();
        let target = seq
            .events
            .iter()
            .position(|event| matches!(event, V4Event::Load(_)))
            .expect("indirect memory target must be modeled before the call");
        let opaque = seq
            .events
            .iter()
            .position(|event| *event == V4Event::OpaqueCall)
            .unwrap();
        assert!(target < opaque);
        let mut insts = vec![Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        }];
        if dirty {
            insts.push(Inst::External {
                inst: asm::inst::movq_mr::new(Amode::imm_reg(8, regs::rdi()), Gpr::RSI).into(),
            });
        }
        insts.push(call);
        let (bytes, _) = selective_native_plan(&insts, V4Policy::Ranges);
        assert_eq!(fences(&bytes), 1 + usize::from(dirty), "{bytes:02x?}");
        assert_eq!(&bytes[bytes.len() - 2..], &[0xff, 0x17]);
        if dirty {
            assert_eq!(
                &bytes[bytes.len() - 5..bytes.len() - 2],
                &[0x0f, 0xae, 0xe8]
            );
        }
    }
}

#[test]
fn selective_call_stack_return_keeps_only_the_real_internal_guard() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    let mut call = selective_call_info(RegMem::Reg { reg: regs::rax() });
    call.defs.push(CallRetPair {
        vreg: Writable::from_reg(regs::rdx()),
        location: RetLocation::Stack(StackAMode::Slot(0), types::I64),
    });
    let insts = [
        Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        },
        Inst::CallUnknown { info: call },
    ];
    let (bytes, _) = selective_native_plan(&insts, V4Policy::Ranges);
    assert_eq!(
        bytes,
        [
            0x0f, 0xae, 0xe8, 0xff, 0xd0, 0x0f, 0xae, 0xe8, 0x48, 0x8b, 0x14, 0x24
        ]
    );
}

#[test]
fn selective_tls_templates_do_not_freeze_clean_entry_guards() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    let symbol = ExternalName::LibCall(LibCall::ElfTlsGetAddr);
    for template in [
        Inst::ElfTlsGetAddr {
            symbol: symbol.clone(),
            dst: Writable::from_reg(Gpr::RAX),
        },
        Inst::MachOTlsGetAddr {
            symbol: symbol.clone(),
            dst: Writable::from_reg(Gpr::RAX),
        },
        Inst::CoffTlsGetAddr {
            symbol,
            dst: Writable::from_reg(Gpr::RAX),
            tmp: Writable::from_reg(Gpr::RCX),
        },
    ] {
        // These linker templates have no store->load inside the expansion.
        // The existing legacy emitter provides an independent exact byte
        // reference for their clean native sequence.
        let reference = emit_raw(&[template.clone()], false);
        assert_eq!(fences(&reference), 0);
        for dirty in [false, true] {
            let mut insts = vec![Inst::External {
                inst: asm::inst::lfence_zo::new().into(),
            }];
            if dirty {
                insts.push(Inst::External {
                    inst: asm::inst::movq_mr::new(Amode::imm_reg(0, regs::rdi()), Gpr::RSI).into(),
                });
            }
            insts.push(template.clone());
            let (bytes, _) = selective_native_plan(&insts, V4Policy::Ranges);
            assert_eq!(
                fences(&bytes),
                1 + usize::from(dirty),
                "{template:?}: {bytes:02x?}"
            );
            assert_eq!(
                &bytes[bytes.len() - reference.len()..],
                reference.as_slice(),
                "the linker template must remain contiguous"
            );
            if dirty {
                assert_eq!(
                    &bytes[bytes.len() - reference.len() - 3..bytes.len() - reference.len()],
                    &[0x0f, 0xae, 0xe8]
                );
            }
        }
    }
}

#[test]
fn selective_tail_argument_template_keeps_internal_guard_without_clean_entry_guard() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    let frame = FrameLayout {
        tail_args_size: 16,
        ..FrameLayout::default()
    };
    let tail = Inst::ReturnCallKnown {
        info: Box::new(ReturnCallInfo {
            dest: ExternalName::LibCall(LibCall::Memcpy),
            new_stack_arg_size: 0,
            uses: smallvec![],
            tmp: Writable::from_reg(Gpr::R11),
        }),
    };
    for dirty in [false, true] {
        let mut insts = vec![Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        }];
        if dirty {
            insts.push(Inst::External {
                inst: asm::inst::movq_mr::new(Amode::imm_reg(0, regs::rdi()), Gpr::RSI).into(),
            });
        }
        insts.push(tail.clone());
        let (bytes, _) = selective_native_plan_with_frame(&insts, V4Policy::Ranges, frame.clone());
        assert_eq!(fences(&bytes), 2 + usize::from(dirty), "{bytes:02x?}");
        assert_eq!(
            &bytes[bytes.len() - 8..],
            &[0x0f, 0xae, 0xe8, 0xe9, 0, 0, 0, 0]
        );
        assert!(
            bytes.windows(3).any(|bytes| bytes == [0x4c, 0x8b, 0x1c]),
            "tail template must still read the native return address"
        );
    }
}

#[test]
fn selective_clean_atomic_entry_keeps_retry_backedge_guard() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    let atomic = Inst::AtomicRmwSeq {
        ty: types::I64,
        op: AtomicRmwSeqOp::Nand,
        mem: SyntheticAmode::real(Amode::imm_reg(0, regs::rdi())),
        operand: Gpr::RSI,
        temp: Writable::from_reg(Gpr::RDX),
        dst_old: Writable::from_reg(Gpr::RAX),
    };
    for dirty in [false, true] {
        let mut insts = vec![Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        }];
        if dirty {
            insts.push(Inst::External {
                inst: asm::inst::movq_mr::new(Amode::imm_reg(8, regs::rdi()), Gpr::RSI).into(),
            });
        }
        if !dirty {
            insts.push(Inst::External {
                inst: asm::inst::movq_rm::new(
                    Writable::from_reg(Gpr::R8),
                    Amode::imm_reg(8, regs::rdi()),
                )
                .into(),
            });
        }
        insts.push(atomic.clone());
        let (bytes, _) = selective_native_plan(&insts, V4Policy::Ranges);
        assert_eq!(fences(&bytes), 2 + usize::from(dirty), "{bytes:02x?}");
        let cas = bytes
            .windows(4)
            .position(|bytes| bytes == [0xf0, 0x48, 0x0f, 0xb1])
            .unwrap();
        assert_eq!(&bytes[cas - 3..cas], &[0x0f, 0xae, 0xe8]);
        let branch = bytes
            .windows(2)
            .position(|bytes| bytes == [0x0f, 0x85])
            .unwrap();
        let displacement = i32::from_le_bytes(bytes[branch + 2..branch + 6].try_into().unwrap());
        let target = (branch as i32 + 6 + displacement) as usize;
        assert!(
            target <= cas - 3 && target > 3,
            "retry must execute its internal guard: {bytes:02x?}"
        );
    }
}

#[test]
fn selective_try_call_cut_and_native_return_load_guard_stay_after_actual_call() {
    use super::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy};
    let info = raw_info();
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.v4_set_selective(true);
    let mut sink = MachBuffer::<Inst>::new();
    let continuation = sink.get_label();
    let handler = sink.get_label();
    let mut call = selective_call_info(ExternalName::LibCall(LibCall::Memcpy));
    call.defs.push(CallRetPair {
        vreg: Writable::from_reg(regs::rdx()),
        location: RetLocation::Stack(StackAMode::Slot(0), types::I64),
    });
    call.try_call_info = Some(TryCallInfo {
        continuation,
        exception_handlers: vec![TryCallHandler::Default(handler)].into_boxed_slice(),
    });
    let inst = Inst::CallKnown { info: call };
    let seq = super::ssbb::classify_v4(
        &inst,
        state.frame_layout(),
        &info,
        &mut V4Context::new(0, V4Policy::Ranges),
    )
    .unwrap();
    let cut = seq.exn_cut.unwrap();
    assert_eq!(seq.events[cut], V4Event::OpaqueCall);
    assert!(
        seq.events
            .iter()
            .skip(cut + 1)
            .any(|event| matches!(event, V4Event::Load(_)))
    );
    state.ssbb_emit_barrier(&mut sink, &info);
    state.v4_set_legacy_internal(seq.legacy_internal);
    inst.emit(&mut sink, &info, &mut state);
    state.v4_set_legacy_internal(false);
    sink.bind_label(handler, state.ctrl_plane_mut());
    let handler_offset = sink.cur_offset();
    state.ssbb_emit_barrier(&mut sink, &info);
    Inst::External {
        inst: asm::inst::retq_zo::new().into(),
    }
    .emit(&mut sink, &info, &mut state);
    sink.bind_label(continuation, state.ctrl_plane_mut());
    Inst::External {
        inst: asm::inst::retq_zo::new().into(),
    }
    .emit(&mut sink, &info, &mut state);
    let code = sink.finish(&Default::default(), &mut ControlPlane::default());
    assert_eq!(
        &code.data()[..15],
        &[
            0x0f, 0xae, 0xe8, 0xe8, 0, 0, 0, 0, 0x0f, 0xae, 0xe8, 0x48, 0x8b, 0x14, 0x24
        ]
    );
    assert_eq!(code.call_sites().next().unwrap().ret_addr, 8);
    assert_eq!(
        code.call_sites().next().unwrap().exception_handlers,
        &[crate::machinst::FinalizedMachExceptionHandler::Default(
            handler_offset
        )]
    );
    assert_eq!(fences(code.data()), 3);
}

#[test]
fn selective_memory_try_call_cut_includes_target_read_and_actual_call() {
    use super::*;
    use crate::machinst::v4_fences::{V4Context, V4Event, V4Policy};
    let info = raw_info();
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.v4_set_selective(true);
    let mut sink = MachBuffer::<Inst>::new();
    let continuation = sink.get_label();
    let handler = sink.get_label();
    let mut call = selective_call_info(RegMem::Mem {
        addr: SyntheticAmode::real(Amode::imm_reg(0, regs::rdi())),
    });
    call.try_call_info = Some(TryCallInfo {
        continuation,
        exception_handlers: vec![TryCallHandler::Default(handler)].into_boxed_slice(),
    });
    let inst = Inst::CallUnknown { info: call };
    let seq = super::ssbb::classify_v4(
        &inst,
        state.frame_layout(),
        &info,
        &mut V4Context::new(0, V4Policy::Ranges),
    )
    .unwrap();
    assert!(matches!(seq.events[0], V4Event::Load(_)));
    assert_eq!(seq.exn_cut, Some(1));
    assert_eq!(seq.events[1], V4Event::OpaqueCall);
    state.ssbb_emit_barrier(&mut sink, &info);
    state.v4_set_legacy_internal(seq.legacy_internal);
    inst.emit(&mut sink, &info, &mut state);
    state.v4_set_legacy_internal(false);
    sink.bind_label(handler, state.ctrl_plane_mut());
    let handler_offset = sink.cur_offset();
    state.ssbb_emit_barrier(&mut sink, &info);
    Inst::External {
        inst: asm::inst::retq_zo::new().into(),
    }
    .emit(&mut sink, &info, &mut state);
    sink.bind_label(continuation, state.ctrl_plane_mut());
    state.ssbb_emit_barrier(&mut sink, &info);
    Inst::External {
        inst: asm::inst::retq_zo::new().into(),
    }
    .emit(&mut sink, &info, &mut state);
    let code = sink.finish(&Default::default(), &mut ControlPlane::default());
    assert_eq!(&code.data()[..5], &[0x0f, 0xae, 0xe8, 0xff, 0x17]);
    assert_eq!(code.call_sites().next().unwrap().ret_addr, 5);
    assert_eq!(
        code.call_sites().next().unwrap().exception_handlers,
        &[crate::machinst::FinalizedMachExceptionHandler::Default(
            handler_offset
        )]
    );
    assert_eq!(fences(code.data()), 3);
}

#[test]
fn selective_disjoint_load_keeps_pending_for_later_aliasing_folded_load() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    let insts = [
        Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        },
        Inst::External {
            inst: asm::inst::movq_mr::new(Amode::imm_reg(0, regs::rdi()), Gpr::RSI).into(),
        },
        Inst::External {
            inst: asm::inst::movzbq_rm::new(
                Writable::from_reg(Gpr::RAX),
                Amode::imm_reg(16, regs::rdi()),
            )
            .into(),
        },
        Inst::External {
            inst: asm::inst::addq_rm::new(
                Writable::from_reg(Gpr::RAX),
                Amode::imm_reg(4, regs::rdi()),
            )
            .into(),
        },
    ];
    let (bytes, plan) = selective_native_plan(&insts, V4Policy::Ranges);
    assert_eq!(plan, [false, false, false, true]);
    assert_eq!(
        bytes,
        [
            0x0f, 0xae, 0xe8, 0x48, 0x89, 0x37, 0x48, 0x0f, 0xb6, 0x47, 0x10, 0x0f, 0xae, 0xe8,
            0x48, 0x03, 0x47, 0x04
        ]
    );
    assert_eq!(
        selective_native_plan(&insts, V4Policy::Frame).1,
        [false, false, true, false]
    );
}

#[test]
fn selective_partial_write_and_reload_invalidate_store_address_proofs() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    for redefine in [
        Inst::External {
            inst: asm::inst::movb_mr::new(Writable::from_reg(Gpr::RDI), Gpr::RSI).into(),
        },
        Inst::External {
            inst: asm::inst::movq_rm::new(
                Writable::from_reg(Gpr::RDI),
                Amode::imm_reg(0, regs::rsp()),
            )
            .into(),
        },
    ] {
        let insts = [
            Inst::External {
                inst: asm::inst::lfence_zo::new().into(),
            },
            Inst::External {
                inst: asm::inst::movq_mr::new(Amode::imm_reg(0, regs::rdi()), Gpr::RSI).into(),
            },
            redefine,
            Inst::External {
                inst: asm::inst::movq_rm::new(
                    Writable::from_reg(Gpr::RAX),
                    Amode::imm_reg(16, regs::rdi()),
                )
                .into(),
            },
        ];
        let (_, plan) = selective_native_plan(&insts, V4Policy::Ranges);
        assert!(
            plan[2] || plan[3],
            "redefinition lost pending store: {plan:?}"
        );
    }
}

#[test]
fn selective_native_frame_ranges_cover_sp_adjustment_and_return_load() {
    use super::*;
    use crate::machinst::v4_fences::V4Policy;
    let insts = [
        Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        },
        Inst::subq_mi(Writable::from_reg(regs::rsp()), 32),
        Inst::External {
            inst: asm::inst::movq_mr::new(Amode::imm_reg(0, regs::rsp()), Gpr::RSI).into(),
        },
        Inst::External {
            inst: asm::inst::movq_rm::new(
                Writable::from_reg(Gpr::RAX),
                Amode::imm_reg(16, regs::rsp()),
            )
            .into(),
        },
        Inst::addq_mi(Writable::from_reg(regs::rsp()), 32),
        Inst::External {
            inst: asm::inst::retq_zo::new().into(),
        },
    ];
    let (_, plan) = selective_native_plan(&insts, V4Policy::Frame);
    assert_eq!(
        plan, [false; 6],
        "fixed frame bytes are disjoint from the return address"
    );
    let insts = [
        Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        },
        Inst::External {
            inst: asm::inst::movq_mr::new(Amode::imm_reg(0, regs::rsp()), Gpr::RSI).into(),
        },
        Inst::External {
            inst: asm::inst::retq_zo::new().into(),
        },
    ];
    assert_eq!(
        selective_native_plan(&insts, V4Policy::Frame).1,
        [false, false, true]
    );
}

fn store_load() -> Function {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params
        .extend([AbiParam::new(types::I64), AbiParam::new(types::I64)]);
    sig.returns.push(AbiParam::new(types::I64));
    let mut f = Function::with_name_signature(UserFuncName::user(0, 0), sig);
    let block = f.dfg.make_block();
    let ptr = f.dfg.append_block_param(block, types::I64);
    let val = f.dfg.append_block_param(block, types::I64);
    let mut pos = FuncCursor::new(&mut f);
    pos.insert_block(block);
    pos.ins().store(MemFlags::trusted(), val, ptr, 0);
    pos.ins().store(MemFlags::trusted(), val, ptr, 8);
    let a = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
    let b = pos.ins().load(types::I64, MemFlags::trusted(), ptr, 8);
    let sum = pos.ins().iadd(a, b);
    pos.ins().return_(&[sum]);
    f
}

fn fences(bytes: &[u8]) -> usize {
    bytes
        .windows(3)
        .filter(|b| *b == [0x0f, 0xae, 0xe8])
        .count()
}

#[test]
fn lfence_coalesces_store_store_load_load() {
    let bytes = compile(store_load(), Some("lfence"));
    assert_eq!(fences(&bytes), 1, "{bytes:02x?}");
}

#[test]
fn lfence_none_matches_default() {
    let default = compile(store_load(), None);
    let none = compile(store_load(), Some("none"));
    assert_eq!(default, none);
    assert_eq!(fences(&none), 0);
}

#[test]
fn lfence_folded_load_and_lea_do_not_lose_store_state() {
    use super::*;
    let flags = settings::Flags::new(settings::builder());
    let info = EmitInfo::new(
        flags,
        crate::isa::x64::settings::Flags::new(
            &settings::Flags::new(settings::builder()),
            &crate::isa::x64::settings::builder(),
        ),
    );
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.ssbb_set_dirty(false);
    let mut sink = MachBuffer::<Inst>::new();
    let rax = Writable::from_reg(Gpr::RAX);
    let memory = || Amode::imm_reg(0, regs::rdi());
    for inst in [
        asm::inst::movq_mr::new(memory(), Gpr::RSI).into(),
        asm::inst::leaq_rm::new(rax, memory()).into(),
        asm::inst::addq_rm::new(rax, memory()).into(),
        asm::inst::movq_rm::new(rax, memory()).into(),
    ] {
        Inst::External { inst }.emit(&mut sink, &info, &mut state);
    }
    let bytes = sink
        .finish(&Default::default(), &mut ControlPlane::default())
        .data()
        .to_vec();
    assert_eq!(
        bytes,
        [
            0x48, 0x89, 0x37, 0x48, 0x8d, 0x07, 0x0f, 0xae, 0xe8, 0x48, 0x03, 0x07, 0x48, 0x8b,
            0x07
        ]
    );
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

#[test]
fn lfence_join_and_loop_back_edge() {
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
    let join = compile(func, Some("lfence"));
    assert_eq!(fences(&join), 2);

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
    let looped = compile(func, Some("lfence"));
    assert_eq!(fences(&looped), 2);
}

fn raw_info() -> super::EmitInfo {
    let mut shared = settings::builder();
    shared.set("preserve_frame_pointers", "true").unwrap();
    let flags = settings::Flags::new(shared);
    let mut isa_flags = crate::isa::x64::settings::builder();
    isa_flags.set("has_cmpxchg16b", "true").unwrap();
    let isa_flags = crate::isa::x64::settings::Flags::new(&flags, &isa_flags);
    super::EmitInfo::new(flags, isa_flags)
}

fn emit_raw(insts: &[super::Inst], dirty: bool) -> Vec<u8> {
    use super::*;
    let info = raw_info();
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.ssbb_set_dirty(dirty);
    let mut sink = MachBuffer::<Inst>::new();
    for inst in insts {
        inst.emit(&mut sink, &info, &mut state);
    }
    sink.finish(&Default::default(), &mut ControlPlane::default())
        .data()
        .to_vec()
}

#[test]
fn lfence_implicit_stack_accesses_and_explicit_fence() {
    use super::*;
    let insts = [
        Inst::External {
            inst: asm::inst::pushq_o::new(Gpr::RBP).into(),
        },
        Inst::External {
            inst: asm::inst::popq_o::new(Writable::from_reg(Gpr::RBP)).into(),
        },
        Inst::External {
            inst: asm::inst::retq_zo::new().into(),
        },
    ];
    assert_eq!(
        emit_raw(&insts, false),
        [0x55, 0x0f, 0xae, 0xe8, 0x5d, 0xc3]
    );
    let insts = [
        Inst::External {
            inst: asm::inst::pushq_o::new(Gpr::RBP).into(),
        },
        Inst::External {
            inst: asm::inst::lfence_zo::new().into(),
        },
        Inst::External {
            inst: asm::inst::popq_o::new(Writable::from_reg(Gpr::RBP)).into(),
        },
    ];
    assert_eq!(emit_raw(&insts, true), [0x55, 0x0f, 0xae, 0xe8, 0x5d]);
}

#[test]
fn lfence_atomic_retry_targets_guard_and_post_atomic_load_is_guarded() {
    use super::*;
    let insts = [
        Inst::AtomicRmwSeq {
            ty: types::I64,
            op: AtomicRmwSeqOp::Nand,
            mem: SyntheticAmode::real(Amode::imm_reg(0, regs::rdi())),
            operand: Gpr::RSI,
            temp: Writable::from_reg(Gpr::RDX),
            dst_old: Writable::from_reg(Gpr::RAX),
        },
        Inst::External {
            inst: asm::inst::movq_rm::new(
                Writable::from_reg(Gpr::RAX),
                Amode::imm_reg(8, regs::rdi()),
            )
            .into(),
        },
    ];
    let bytes = emit_raw(&insts, false);
    assert_eq!(fences(&bytes), 2, "{bytes:02x?}");
    let cas = bytes
        .windows(4)
        .position(|b| b == [0xf0, 0x48, 0x0f, 0xb1])
        .unwrap();
    assert_eq!(&bytes[cas - 3..cas], &[0x0f, 0xae, 0xe8]);
    let branch = bytes.windows(2).position(|b| b == [0x0f, 0x85]).unwrap();
    let disp = i32::from_le_bytes(bytes[branch + 2..branch + 6].try_into().unwrap());
    let target = (branch as i32 + 6 + disp) as usize;
    assert!(target <= cas - 3, "retry skipped fence: {bytes:02x?}");
    assert_eq!(target, 3, "retry should follow initial MOV: {bytes:02x?}");
    assert_eq!(&bytes[branch + 6..branch + 9], &[0x0f, 0xae, 0xe8]);
}

#[test]
fn lfence_patchable_call_offsets_exclude_guard() {
    use super::*;
    let info = raw_info();
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.ssbb_set_dirty(true);
    let mut sink = MachBuffer::<Inst>::new();
    let call = Inst::CallKnown {
        info: Box::new(CallInfo {
            dest: ExternalName::LibCall(LibCall::Memcpy),
            uses: smallvec![],
            defs: smallvec![],
            clobbers: Default::default(),
            callee_conv: CallConv::SystemV,
            caller_conv: CallConv::SystemV,
            callee_pop_size: 0,
            try_call_info: None,
            patchable: true,
        }),
    };
    state.pre_safepoint(Some(crate::ir::UserStackMap::new(&[], &Default::default())));
    call.emit(&mut sink, &info, &mut state);
    let code = sink.finish(&Default::default(), &mut ControlPlane::default());
    assert_eq!(code.data(), [0x0f, 0xae, 0xe8, 0xe8, 0, 0, 0, 0]);
    assert_eq!(code.relocs()[0].offset, 4);
    assert_eq!(code.user_stack_maps()[0].0, 8);
    assert_eq!(code.call_sites().next().unwrap().ret_addr, 8);
    let patch = code.patchable_call_sites().next().unwrap();
    assert_eq!((patch.ret_addr, patch.len), (8, 5));
}

#[test]
fn lfence_call_stack_returns_are_guarded_inside_expansion() {
    use super::*;
    let call = Inst::CallUnknown {
        info: Box::new(CallInfo {
            dest: RegMem::Reg { reg: regs::rax() },
            uses: smallvec![],
            defs: smallvec![CallRetPair {
                vreg: Writable::from_reg(regs::rdx()),
                location: RetLocation::Stack(StackAMode::Slot(0), types::I64),
            }],
            clobbers: Default::default(),
            callee_conv: CallConv::SystemV,
            caller_conv: CallConv::SystemV,
            callee_pop_size: 0,
            try_call_info: None,
            patchable: false,
        }),
    };
    let bytes = emit_raw(&[call], true);
    assert_eq!(
        bytes,
        [
            0x0f, 0xae, 0xe8, 0xff, 0xd0, 0x0f, 0xae, 0xe8, 0x48, 0x8b, 0x14, 0x24
        ]
    );
}

#[test]
fn lfence_call_and_regalloc_spills_compile() {
    let mut func = new_func(&[types::I64], &[types::I64]);
    let (mut pos, vals) = entry(&mut func, &[types::I64]);
    let callee = import_func(pos.func, Signature::new(CallConv::SystemV));
    let mut live = Vec::new();
    for i in 0..32 {
        live.push(
            pos.ins()
                .load(types::I64, MemFlags::trusted(), vals[0], i * 8),
        );
    }
    pos.ins().call(callee, &[]);
    let mut acc = live[0];
    for v in live.into_iter().skip(1) {
        acc = pos.ins().iadd(acc, v);
    }
    pos.ins().return_(&[acc]);
    for policy in ["frame", "ranges", "ranges_cfg"] {
        let bytes = compile_policy(func.clone(), Some("lfence"), policy);
        assert!(
            fences(&bytes) >= 2,
            "{policy} must protect call and post-call regalloc reloads: {bytes:02x?}"
        );
    }
    let bytes = compile(func, Some("lfence"));
    assert!(
        fences(&bytes) >= 3,
        "entry, store/load spills, and post-call reload must be guarded: {bytes:02x?}"
    );
}

#[test]
fn lfence_presets_do_not_overflow_or_change_mode() {
    let flags = settings::Flags::new(settings::builder());
    let mut builder = crate::isa::x64::settings::builder();
    builder.set("x64_mitigation", "lfence").unwrap();
    for preset in ["znver4", "x86-64-v4"] {
        builder.enable(preset).unwrap();
        let flags = crate::isa::x64::settings::Flags::new(&flags, &builder);
        assert_eq!(
            flags.x64_mitigation(),
            crate::isa::x64::settings::X64Mitigation::Lfence
        );
        assert!(flags.has_avx512f());
    }
}

#[test]
fn lfence_tail_call_argument_move_guards_final_jump() {
    use super::*;
    let info = raw_info();
    let mut state = EmitState::default();
    state.frame_layout.tail_args_size = 16;
    state.ssbb_install(&[]);
    state.ssbb_set_dirty(true);
    let mut sink = MachBuffer::<Inst>::new();
    let inst = Inst::ReturnCallKnown {
        info: Box::new(ReturnCallInfo {
            dest: ExternalName::LibCall(LibCall::Memcpy),
            new_stack_arg_size: 0,
            uses: smallvec![],
            tmp: Writable::from_reg(Gpr::R11),
        }),
    };
    inst.emit(&mut sink, &info, &mut state);
    let code = sink.finish(&Default::default(), &mut ControlPlane::default());
    let bytes = code.data();
    assert_eq!(fences(bytes), 2, "{bytes:02x?}");
    assert_eq!(
        &bytes[bytes.len() - 8..bytes.len() - 4],
        &[0x0f, 0xae, 0xe8, 0xe9]
    );
    assert_eq!(code.relocs()[0].offset as usize, bytes.len() - 4);
}

#[test]
fn lfence_tls_templates_keep_linker_bytes_contiguous() {
    use super::*;
    let symbol = ExternalName::LibCall(LibCall::ElfTlsGetAddr);
    let elf = Inst::ElfTlsGetAddr {
        symbol: symbol.clone(),
        dst: Writable::from_reg(Gpr::RAX),
    };
    let bytes = emit_raw(&[elf], true);
    assert_eq!(
        bytes,
        [
            0x0f, 0xae, 0xe8, 0x66, 0x48, 0x8d, 0x3d, 0, 0, 0, 0, 0x66, 0x66, 0x48, 0xe8, 0, 0, 0,
            0
        ]
    );
    for inst in [
        Inst::MachOTlsGetAddr {
            symbol: symbol.clone(),
            dst: Writable::from_reg(Gpr::RAX),
        },
        Inst::CoffTlsGetAddr {
            symbol,
            dst: Writable::from_reg(Gpr::RAX),
            tmp: Writable::from_reg(Gpr::RCX),
        },
    ] {
        let bytes = emit_raw(&[inst], true);
        assert_eq!(fences(&bytes), 1, "{bytes:02x?}");
        assert_eq!(&bytes[..3], &[0x0f, 0xae, 0xe8]);
    }
}

#[test]
fn lfence_fault_metadata_points_at_load_after_guard() {
    use super::*;
    let info = raw_info();
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.ssbb_set_dirty(true);
    let mut sink = MachBuffer::<Inst>::new();
    let inst = Inst::External {
        inst: asm::inst::movq_rm::new(
            Writable::from_reg(Gpr::RAX),
            Amode::imm_reg(0, regs::rdi()).with_flags(MemFlags::new()),
        )
        .into(),
    };
    inst.emit(&mut sink, &info, &mut state);
    let code = sink.finish(&Default::default(), &mut ControlPlane::default());
    assert_eq!(code.data(), [0x0f, 0xae, 0xe8, 0x48, 0x8b, 0x07]);
    assert_eq!(code.traps()[0].offset, 3);
}

#[test]
fn lfence_unsupported_internal_control_flow_fails_closed() {
    use super::*;
    let info = raw_info();
    for inst in [
        Inst::WinchJmpIf {
            cc: CC::Z,
            taken: MachLabel::from_block(crate::machinst::BlockIndex::new(0)),
        },
        Inst::StackSwitchBasic {
            store_context_ptr: Gpr::RAX,
            load_context_ptr: Gpr::RBX,
            in_payload0: Gpr::RCX,
            out_payload0: Writable::from_reg(Gpr::RDX),
        },
    ] {
        assert!(matches!(
            super::ssbb::classify(&inst, &Default::default(), &info),
            Err(crate::CodegenError::Unsupported(_))
        ));
    }
}

#[test]
fn lfence_atomic128_rmw_and_exchange_retry_include_guard() {
    use super::*;
    use crate::isa::x64::lower::isle::generated_code::Atomic128RmwSeqOp;
    let memory = || SyntheticAmode::real(Amode::imm_reg(0, regs::rdi()));
    let low = Writable::from_reg(Gpr::RAX);
    let high = Writable::from_reg(Gpr::RDX);
    for atomic in [
        Inst::Atomic128XchgSeq {
            mem: memory(),
            operand_low: Gpr::RBX,
            operand_high: Gpr::RCX,
            dst_old_low: low,
            dst_old_high: high,
        },
        Inst::Atomic128RmwSeq {
            mem: Box::new(memory()),
            op: Atomic128RmwSeqOp::Add,
            operand_low: Gpr::RSI,
            operand_high: Gpr::R8,
            temp_low: Writable::from_reg(Gpr::RBX),
            temp_high: Writable::from_reg(Gpr::RCX),
            dst_old_low: low,
            dst_old_high: high,
        },
    ] {
        let load = Inst::External {
            inst: asm::inst::movq_rm::new(low, Amode::imm_reg(0, regs::rdi())).into(),
        };
        let bytes = emit_raw(&[atomic, load], false);
        assert_eq!(fences(&bytes), 2, "{bytes:02x?}");
        let cas = bytes
            .windows(4)
            .position(|b| b == [0xf0, 0x48, 0x0f, 0xc7])
            .unwrap();
        assert_eq!(&bytes[cas - 3..cas], &[0x0f, 0xae, 0xe8]);
        let branch = bytes.windows(2).position(|b| b == [0x0f, 0x85]).unwrap();
        let disp = i32::from_le_bytes(bytes[branch + 2..branch + 6].try_into().unwrap());
        let target = (branch as i32 + 6 + disp) as usize;
        assert_eq!(
            target, 7,
            "retry follows the two initial loads: {bytes:02x?}"
        );
        assert!(target <= cas - 3, "retry skipped fence: {bytes:02x?}");
        assert_eq!(&bytes[branch + 6..branch + 9], &[0x0f, 0xae, 0xe8]);
    }
}

#[test]
fn lfence_try_call_exception_state_precedes_return_load_and_offsets_are_valid() {
    use super::*;
    use crate::machinst::ssbb::{SsbbMemStep, SsbbSeq};
    let info = raw_info();
    let mut state = EmitState::default();
    state.ssbb_install(&[]);
    state.ssbb_set_dirty(true);
    let mut sink = MachBuffer::<Inst>::new();
    let continuation = sink.get_label();
    let handler = sink.get_label();
    let inst = Inst::CallKnown {
        info: Box::new(CallInfo {
            dest: ExternalName::LibCall(LibCall::Memcpy),
            uses: smallvec![],
            defs: smallvec![CallRetPair {
                vreg: Writable::from_reg(regs::rdx()),
                location: RetLocation::Stack(StackAMode::Slot(0), types::I64)
            }],
            clobbers: Default::default(),
            callee_conv: CallConv::SystemV,
            caller_conv: CallConv::SystemV,
            callee_pop_size: 0,
            try_call_info: Some(TryCallInfo {
                continuation,
                exception_handlers: vec![TryCallHandler::Default(handler)].into_boxed_slice(),
            }),
            patchable: false,
        }),
    };
    let seq = super::ssbb::classify(&inst, state.frame_layout(), &info).unwrap();
    assert_eq!(
        seq.steps.as_slice(),
        &[SsbbMemStep::Call, SsbbMemStep::Load]
    );
    assert_eq!(seq.exn_cut, Some(0));
    assert!(SsbbSeq::transfer(
        false,
        &seq.steps[..=seq.exn_cut.unwrap() as usize]
    ));
    assert!(!SsbbSeq::transfer(false, &seq.steps));
    inst.emit(&mut sink, &info, &mut state);
    sink.bind_label(handler, state.ctrl_plane_mut());
    let handler_offset = sink.cur_offset();
    state.ssbb_set_dirty(true);
    Inst::External {
        inst: asm::inst::retq_zo::new().into(),
    }
    .emit(&mut sink, &info, &mut state);
    sink.bind_label(continuation, state.ctrl_plane_mut());
    state.ssbb_set_dirty(false);
    Inst::External {
        inst: asm::inst::retq_zo::new().into(),
    }
    .emit(&mut sink, &info, &mut state);
    let code = sink.finish(&Default::default(), &mut ControlPlane::default());
    let call = code.call_sites().next().unwrap();
    assert_eq!(call.ret_addr, 8);
    assert_eq!(
        call.exception_handlers,
        &[crate::machinst::FinalizedMachExceptionHandler::Default(
            handler_offset
        )]
    );
    assert_eq!(
        &code.data()[handler_offset as usize..handler_offset as usize + 4],
        &[0x0f, 0xae, 0xe8, 0xc3]
    );
    assert_eq!(fences(code.data()), 3);
}
