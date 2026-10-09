use super::*;
use crate::ir;
use cranelift_control::ControlPlane;

/// State carried between emissions of a sequence of instructions.
#[derive(Default, Clone, Debug)]
pub struct EmitState {
    /// The user stack map for the upcoming instruction, as provided to
    /// `pre_safepoint()`.
    user_stack_map: Option<ir::UserStackMap>,

    /// Only used during fuzz-testing. Otherwise, it is a zero-sized struct and
    /// optimized away at compiletime. See [cranelift_control].
    ctrl_plane: ControlPlane,

    /// A copy of the frame layout, used during the emission of `Inst::ReturnCallKnown` and
    /// `Inst::ReturnCallUnknown` instructions and exception callsites.
    pub frame_layout: FrameLayout,

    ssbb_active: bool,
    ssbb_dirty: bool,
    ssbb_barriers: u32,
    // An outer expansion already accounted for all effects; recursive emits
    // must not account for them a second time.
    pub(crate) ssbb_suppressed: bool,
    v4_selective: bool,
    v4_legacy_internal: bool,
}

impl MachInstEmitState<Inst> for EmitState {
    fn new(abi: &Callee<X64ABIMachineSpec>, ctrl_plane: ControlPlane) -> Self {
        EmitState {
            user_stack_map: None,
            ctrl_plane,
            frame_layout: abi.frame_layout().clone(),
            ..Self::default()
        }
    }

    fn ssbb_install(&mut self, _block_in: &[bool]) {
        self.ssbb_active = true;
    }
    fn ssbb_active(&self) -> bool {
        self.ssbb_active
    }
    fn ssbb_set_dirty(&mut self, dirty: bool) {
        if self.ssbb_active {
            self.ssbb_dirty = dirty;
        }
    }
    fn ssbb_dirty(&self) -> bool {
        self.ssbb_active && self.ssbb_dirty
    }
    fn ssbb_reset_barrier_count(&mut self) {
        self.ssbb_barriers = 0;
    }
    fn ssbb_barrier_count(&self) -> u32 {
        self.ssbb_barriers
    }
    fn v4_set_selective(&mut self, active: bool) {
        self.v4_selective = active;
    }
    fn v4_set_legacy_internal(&mut self, active: bool) {
        self.v4_legacy_internal = active;
        if active {
            self.ssbb_dirty = true;
            self.ssbb_suppressed = false;
        }
    }
    fn ssbb_emit_barrier(&mut self, sink: &mut MachBuffer<Inst>, _info: &EmitInfo) {
        sink.put_data(&[0x0f, 0xae, 0xe8]);
        self.ssbb_dirty = false;
        self.ssbb_barriers += 1;
    }

    fn pre_safepoint(&mut self, user_stack_map: Option<ir::UserStackMap>) {
        self.user_stack_map = user_stack_map;
    }

    fn ctrl_plane_mut(&mut self) -> &mut ControlPlane {
        &mut self.ctrl_plane
    }

    fn take_ctrl_plane(self) -> ControlPlane {
        self.ctrl_plane
    }

    fn frame_layout(&self) -> &FrameLayout {
        &self.frame_layout
    }
}

impl EmitState {
    pub(crate) fn ssbb_auto_active(&self) -> bool {
        self.ssbb_active && (!self.v4_selective || self.v4_legacy_internal)
    }
    pub(crate) fn ssbb_step(
        &mut self,
        step: crate::machinst::ssbb::SsbbMemStep,
        sink: &mut MachBuffer<Inst>,
        info: &EmitInfo,
    ) {
        use crate::machinst::ssbb::SsbbSeq;
        if self.ssbb_auto_active() && !self.ssbb_suppressed {
            if SsbbSeq::needs_guard(self.ssbb_dirty, step) {
                self.ssbb_emit_barrier(sink, info);
            }
            self.ssbb_dirty = SsbbSeq::transfer(self.ssbb_dirty, &[step]);
        }
    }

    pub(crate) fn take_stack_map(&mut self) -> Option<ir::UserStackMap> {
        self.user_stack_map.take()
    }

    pub(crate) fn clear_post_insn(&mut self) {
        self.user_stack_map = None;
    }
}
