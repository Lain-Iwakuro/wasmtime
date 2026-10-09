#![allow(missing_docs)]
//! Postregalloc native memory effects and independently checked fence plans.

use super::Reg;
use super::ssbb::{SsbbMemStep, SsbbSeq};
use alloc::vec::Vec;
use alloc::{format, vec};
use smallvec::SmallVec;

pub const RANGE_LIMIT: usize = 32;
pub const SOLVE_LIMIT: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V4Policy {
    Frame,
    Ranges,
    RangesCfg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V4Range {
    Unknown,
    Frame {
        frame: u32,
        offset: i64,
        width: u32,
    },
    LocalBase {
        block: u32,
        reg: Reg,
        epoch: u32,
        offset: i64,
        width: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V4BarrierOrigin {
    Upstream,
    LegacyInternal,
    Experiment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V4Event {
    Pure,
    RegisterDef(Reg),
    Load(V4Range),
    Store(V4Range),
    Rmw(V4Range, V4Range),
    Barrier(V4BarrierOrigin),
    OpaqueCall,
}

#[derive(Clone, Debug)]
pub struct V4Seq {
    pub events: SmallVec<[V4Event; 8]>,
    /// Inclusive exception cut; last successor uses the full sequence.
    pub exn_cut: Option<usize>,
    /// This expansion emits its unchanged conservative internal guards.
    /// Its emitter starts clean; the containing instruction's optional guard
    /// handles incoming Pending, while only actual internal guards are frozen.
    pub legacy_internal: bool,
}

impl V4Seq {
    pub fn neutral() -> Self {
        Self {
            events: SmallVec::new(),
            exn_cut: None,
            legacy_internal: false,
        }
    }

    /// Preserve legacy guards required after the expansion has started. The
    /// first read remains visible to the checker as an external guard site.
    pub fn frozen_legacy(seq: &SsbbSeq) -> Self {
        let mut result = Self::neutral();
        result.legacy_internal = true;
        let mut dirty = false;
        for (i, step) in seq.steps.iter().copied().enumerate() {
            if SsbbSeq::needs_guard(dirty, step) {
                result
                    .events
                    .push(V4Event::Barrier(V4BarrierOrigin::LegacyInternal));
            }
            result.events.push(match step {
                SsbbMemStep::Load => V4Event::Load(V4Range::Unknown),
                SsbbMemStep::Store => V4Event::Store(V4Range::Unknown),
                SsbbMemStep::Atomic => V4Event::Rmw(V4Range::Unknown, V4Range::Unknown),
                SsbbMemStep::Call => V4Event::OpaqueCall,
                SsbbMemStep::Clear => V4Event::Barrier(V4BarrierOrigin::Upstream),
            });
            if seq.exn_cut == Some(i as u16) {
                result.exn_cut = Some(result.events.len() - 1);
            }
            dirty = SsbbSeq::transfer(dirty, &[step]);
        }
        result
    }

    pub fn fixed_barriers(&self) -> u32 {
        self.events
            .iter()
            .filter(|event| matches!(event, V4Event::Barrier(V4BarrierOrigin::LegacyInternal)))
            .count() as u32
    }
}

/// Sequential classifier facts. Registers use canonical physical identities;
/// partial-register writes and W/X writes must have the same `Reg` identity.
#[derive(Clone, Debug)]
pub struct V4Context {
    pub block: u32,
    pub policy: V4Policy,
    pub external_entry: bool,
    epochs: Vec<(Reg, u32)>,
    frame_offsets: Vec<(Reg, i64)>,
}

impl V4Context {
    pub fn new(block: u32, policy: V4Policy) -> Self {
        Self {
            block,
            policy,
            external_entry: false,
            epochs: Vec::new(),
            frame_offsets: Vec::new(),
        }
    }
    pub fn start_block(&mut self, block: u32) {
        self.block = block;
        self.external_entry = false;
        self.epochs.clear();
    }
    pub fn frame_offset(&self, reg: Reg) -> Option<i64> {
        self.frame_offsets
            .iter()
            .find(|(r, _)| *r == reg)
            .map(|(_, offset)| *offset)
    }
    pub fn frame_facts(&self) -> &[(Reg, i64)] {
        &self.frame_offsets
    }

    pub fn clear_frame_facts(&mut self) {
        self.frame_offsets.clear();
    }
    pub fn set_frame_offset(&mut self, reg: Reg, offset: Option<i64>) {
        self.frame_offsets.retain(|(r, _)| *r != reg);
        if let Some(offset) = offset {
            self.frame_offsets.push((reg, offset));
        }
    }
    pub fn adjust_frame_offset(&mut self, reg: Reg, delta: i64) {
        let offset = self
            .frame_offset(reg)
            .and_then(|offset| offset.checked_add(delta));
        self.set_frame_offset(reg, offset);
    }
    pub fn frame_range(&self, reg: Reg, offset: i64, width: u32) -> V4Range {
        self.frame_offset(reg)
            .and_then(|base| base.checked_add(offset))
            .map(|offset| V4Range::Frame {
                frame: 0,
                offset,
                width,
            })
            .unwrap_or(V4Range::Unknown)
    }
    pub fn local_base(&self, reg: Reg, offset: i64, width: u32) -> V4Range {
        if self.policy == V4Policy::Frame || reg.to_real_reg().is_none() {
            return V4Range::Unknown;
        }
        let epoch = self
            .epochs
            .iter()
            .find(|(r, _)| *r == reg)
            .map(|(_, epoch)| *epoch)
            .unwrap_or(0);
        V4Range::LocalBase {
            block: self.block,
            reg,
            epoch,
            offset,
            width,
        }
    }
    /// Call after classifying every old effective address. The caller must also
    /// append `RegisterDef(reg)` to the event sequence for checker invalidation.
    pub fn def(&mut self, reg: Reg) {
        self.set_frame_offset(reg, None);
        if let Some((_, epoch)) = self.epochs.iter_mut().find(|(r, _)| *r == reg) {
            *epoch = epoch.saturating_add(1);
        } else {
            self.epochs.push((reg, 1));
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V4EventId {
    pub block: u32,
    pub inst: u32,
    pub sub: u16,
}

#[derive(Clone, Debug)]
pub struct V4Node {
    pub id: V4EventId,
    pub event: V4Event,
    /// Legal fence position before the containing machine instruction.
    pub guard: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct V4Edge {
    pub target: usize,
    /// Inclusive event cut for exceptional successors, or normal block exit.
    pub cut: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct V4Block {
    pub events: Vec<V4Node>,
    pub successors: Vec<V4Edge>,
    /// Backend-proven legal position before a pure unconditional branch.
    /// The preceding original instruction must emit actual non-branch code,
    /// so a guard cannot interrupt MachBuffer branch folding or threading.
    pub exit_site: Option<usize>,
    pub loop_depth: usize,
}

#[derive(Clone, Debug)]
pub struct V4Graph {
    pub blocks: Vec<V4Block>,
    pub entry: usize,
    pub sites: usize,
}

/// Actual historical guard sites, replayed from original `SsbbSeq`s and the
/// original `SsbbPlan` block states, never regenerated on the selective graph.
#[derive(Clone, Debug)]
pub struct V4Baseline {
    pub fences: Vec<bool>,
    pub fixed_internal_fences: usize,
}

/// Replay one original instruction. The only difference between dirty and
/// clean entry can be a single external guard before the first read. All other
/// historical guards must match the fixed internal selective events exactly.
pub fn replay_legacy_instruction(
    sequence: &SsbbSeq,
    effects: &V4Seq,
    dirty: &mut bool,
) -> crate::CodegenResult<bool> {
    let internal = SsbbSeq::count_barriers(false, &sequence.steps);
    let actual = SsbbSeq::count_barriers(*dirty, &sequence.steps);
    if internal != effects.fixed_barriers() || actual < internal || actual - internal > 1 {
        return Err(crate::CodegenError::Unsupported(format!(
            "Spectre-v4 actual legacy guard mismatch: old={sequence:?}, entry_dirty={dirty}, clean_internal={internal}, actual={actual}, selective={effects:?}"
        )));
    }
    *dirty = SsbbSeq::transfer(*dirty, &sequence.steps);
    Ok(actual > internal)
}

#[derive(Clone, Debug)]
pub struct V4CfgMove {
    pub removed_site: usize,
    pub added_sites: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct V4Plan {
    pub fences: Vec<bool>,
    pub legacy_fences: usize,
    pub solve_count: usize,
    pub fallback: Option<&'static str>,
    pub moved: usize,
    pub cfg_moves: Vec<V4CfgMove>,
    pub cfg_added_sites: Vec<usize>,
    pub cfg_removed_sites: Vec<usize>,
    pub baseline_added_sites: Vec<usize>,
    pub baseline_removed_sites: Vec<usize>,
    pub unknown_reasons: Vec<UnknownReason>,
}

/// Low twelve-bit sufficient condition; no expression or register-name alias
/// assumptions and no same-address exemption are made here.
pub fn no_overlap(a: V4Range, b: V4Range) -> bool {
    match (a, b) {
        (
            V4Range::Frame {
                frame: af,
                offset: ao,
                width: aw,
            },
            V4Range::Frame {
                frame: bf,
                offset: bo,
                width: bw,
            },
        ) if af == bf && aw > 0 && bw > 0 => {
            if ao.checked_add(i64::from(aw)).is_none() || bo.checked_add(i64::from(bw)).is_none() {
                return false;
            }
            let a0 = i128::from(ao);
            let b0 = i128::from(bo);
            a0 + i128::from(aw) <= b0 || b0 + i128::from(bw) <= a0
        }
        (
            V4Range::LocalBase {
                block: ab,
                reg: ar,
                epoch: ae,
                offset: ao,
                width: aw,
            },
            V4Range::LocalBase {
                block: bb,
                reg: br,
                epoch: be,
                offset: bo,
                width: bw,
            },
        ) if (ab, ar, ae) == (bb, br, be)
            && (1..=4096).contains(&aw)
            && (1..=4096).contains(&bw) =>
        {
            let intervals = |offset: i64, width: u32| {
                let start = offset.rem_euclid(4096) as u32;
                let end = start + width;
                [(start, end.min(4096)), (0, end.saturating_sub(4096))]
            };
            intervals(ao, aw).iter().all(|&(a0, a1)| {
                intervals(bo, bw)
                    .iter()
                    .all(|&(b0, b1)| a0 == a1 || b0 == b1 || a1 <= b0 || b1 <= a0)
            })
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    ExternalEntry,
    UnknownAddress,
    RegisterRedefined,
    BlockBoundary,
    RangeLimit,
    OpaqueCall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V4Store {
    pub range: V4Range,
    pub source: V4EventId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pending {
    Empty,
    Known(Vec<V4Store>),
    Unknown {
        sources: Vec<V4EventId>,
        reasons: Vec<UnknownReason>,
    },
}

fn add_unique<T: Copy + PartialEq>(into: &mut Vec<T>, item: T, limit: usize) {
    if !into.contains(&item) && into.len() < limit {
        into.push(item);
    }
}

impl Pending {
    fn unknown(reason: UnknownReason, source: Option<V4EventId>) -> Self {
        Self::Unknown {
            sources: source.into_iter().collect(),
            reasons: vec![reason],
        }
    }
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
    fn promote(&mut self, reason: UnknownReason) {
        match self {
            Self::Empty => {}
            Self::Known(stores) => {
                *self = Self::Unknown {
                    sources: stores.iter().map(|s| s.source).collect(),
                    reasons: vec![reason],
                };
            }
            Self::Unknown { reasons, .. } => add_unique(reasons, reason, 8),
        }
    }
    fn add_store(&mut self, range: V4Range, source: V4EventId) {
        if range == V4Range::Unknown {
            self.join(&Self::unknown(UnknownReason::UnknownAddress, Some(source)));
            return;
        }
        match self {
            Self::Empty => *self = Self::Known(vec![V4Store { range, source }]),
            Self::Known(stores) => {
                if !stores.iter().any(|s| s.range == range) {
                    if stores.len() == RANGE_LIMIT {
                        self.promote(UnknownReason::RangeLimit);
                    } else {
                        stores.push(V4Store { range, source });
                    }
                }
            }
            Self::Unknown { sources, .. } => add_unique(sources, source, RANGE_LIMIT),
        }
    }
    fn join(&mut self, other: &Self) -> bool {
        let before = self.clone();
        match other {
            Self::Empty => {}
            Self::Known(stores) => {
                for store in stores {
                    self.add_store(store.range, store.source);
                }
            }
            Self::Unknown { sources, reasons } => {
                if !matches!(self, Self::Unknown { .. }) {
                    let old_sources = match self {
                        Self::Known(stores) => stores.iter().map(|s| s.source).collect(),
                        _ => Vec::new(),
                    };
                    *self = Self::Unknown {
                        sources: old_sources,
                        reasons: Vec::new(),
                    };
                }
                if let Self::Unknown {
                    sources: ours,
                    reasons: our_reasons,
                } = self
                {
                    for source in sources {
                        add_unique(ours, *source, RANGE_LIMIT);
                    }
                    for reason in reasons {
                        add_unique(our_reasons, *reason, 8);
                    }
                }
            }
        }
        *self != before
    }
    fn crossed_block(mut self) -> Self {
        if matches!(&self, Self::Known(stores) if stores.iter().any(|s| matches!(s.range, V4Range::LocalBase { .. })))
        {
            self.promote(UnknownReason::BlockBoundary);
        }
        self
    }
    fn permits(&self, load: V4Range) -> bool {
        match self {
            Self::Empty => true,
            Self::Known(stores) => stores.iter().all(|store| no_overlap(store.range, load)),
            Self::Unknown { .. } => false,
        }
    }
    fn apply(&mut self, event: V4Event, id: V4EventId) {
        match event {
            V4Event::Pure | V4Event::Load(_) => {}
            V4Event::RegisterDef(reg) => {
                if matches!(&self, Self::Known(stores) if stores.iter().any(|s| matches!(s.range, V4Range::LocalBase { reg: r, .. } if r == reg)))
                {
                    self.promote(UnknownReason::RegisterRedefined);
                }
            }
            V4Event::Store(range) | V4Event::Rmw(_, range) => self.add_store(range, id),
            V4Event::Barrier(_) => *self = Self::Empty,
            V4Event::OpaqueCall => *self = Self::unknown(UnknownReason::OpaqueCall, Some(id)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct V4CoverageError {
    pub block: usize,
    pub event: V4EventId,
    pub load: V4Range,
    pub store_sources: Vec<V4EventId>,
    pub unknown_reasons: Vec<UnknownReason>,
    /// A reproducible CFG path from the external entry to this block. Store
    /// witnesses additionally identify the unsafe abstract predecessor effects.
    pub path: Vec<usize>,
    pub guard: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct V4Check {
    pub errors: Vec<V4CoverageError>,
    pub block_in: Vec<Pending>,
    pub edge_out: Vec<Vec<Pending>>,
    pub unknown_reasons: Vec<UnknownReason>,
}

fn transfer_block(block: &V4Block, incoming: Pending, fences: &[bool]) -> (Pending, Vec<Pending>) {
    let mut pending = incoming;
    let mut snapshots = Vec::with_capacity(block.events.len());
    let mut previous_site = None;
    for node in &block.events {
        if node.guard != previous_site && node.guard.is_some_and(|site| fences[site]) {
            pending = Pending::Empty;
        }
        previous_site = node.guard;
        pending.apply(node.event, node.id);
        snapshots.push(pending.clone());
    }
    let edges = block
        .successors
        .iter()
        .map(|edge| {
            edge.cut
                .map(|cut| snapshots[cut].clone())
                .unwrap_or_else(|| pending.clone())
                .crossed_block()
        })
        .collect();
    (pending, edges)
}

fn path_to(graph: &V4Graph, target: usize) -> Vec<usize> {
    let mut previous = vec![None; graph.blocks.len()];
    previous[graph.entry] = Some(graph.entry);
    let mut queue = vec![graph.entry];
    let mut cursor = 0;
    while cursor < queue.len() {
        let block = queue[cursor];
        cursor += 1;
        for edge in &graph.blocks[block].successors {
            if previous[edge.target].is_none() {
                previous[edge.target] = Some(block);
                queue.push(edge.target);
            }
        }
    }
    if previous[target].is_none() {
        return vec![target];
    }
    let mut path = vec![target];
    let mut block = target;
    while block != graph.entry {
        block = previous[block].unwrap();
        path.push(block);
    }
    path.reverse();
    path
}

fn store_path(
    graph: &V4Graph,
    goal: usize,
    target: V4EventId,
    sources: &[V4EventId],
    edge_out: &[Vec<Pending>],
) -> Vec<usize> {
    let Some(source) = sources.first().copied() else {
        return path_to(graph, goal);
    };
    let Some(start) = graph
        .blocks
        .iter()
        .position(|block| block.events.iter().any(|node| node.id == source))
    else {
        return path_to(graph, goal);
    };
    let mut prefix = path_to(graph, start);
    if start == goal && (source.inst, source.sub) < (target.inst, target.sub) {
        return prefix;
    }
    let mut previous = vec![None; graph.blocks.len()];
    let mut queue = vec![start];
    let mut cursor = 0;
    while cursor < queue.len() {
        let block = queue[cursor];
        cursor += 1;
        for (i, edge) in graph.blocks[block].successors.iter().enumerate() {
            let carries = match &edge_out[block][i] {
                Pending::Known(stores) => stores.iter().any(|store| store.source == source),
                Pending::Unknown { sources, .. } => sources.contains(&source),
                Pending::Empty => false,
            };
            if !carries || previous[edge.target].is_some() {
                continue;
            }
            previous[edge.target] = Some(block);
            if edge.target == goal {
                let mut suffix = vec![goal];
                let mut walk = block;
                while walk != start {
                    suffix.push(walk);
                    walk = previous[walk].unwrap();
                }
                suffix.reverse();
                prefix.extend(suffix);
                return prefix;
            }
            queue.push(edge.target);
        }
    }
    // Abstract Unknown promotion can merge several witnesses; keep the store
    // prefix and explicit source IDs even if that one witness was coalesced.
    path_to(graph, goal)
}

/// Check a fixed graph and fixed fence set. This function never adds a fence;
/// only its converged fixed point is used to report coverage violations.
pub fn check(graph: &V4Graph, fences: &[bool]) -> V4Check {
    assert_eq!(fences.len(), graph.sites);
    let n = graph.blocks.len();
    let mut incoming = vec![Pending::Empty; n];
    incoming[graph.entry] = Pending::unknown(UnknownReason::ExternalEntry, None);
    let mut work: Vec<usize> = (0..n).rev().collect();
    let mut queued = vec![true; n];
    while let Some(block) = work.pop() {
        queued[block] = false;
        let (_, edges) = transfer_block(&graph.blocks[block], incoming[block].clone(), fences);
        for (edge, outgoing) in graph.blocks[block].successors.iter().zip(edges) {
            if incoming[edge.target].join(&outgoing) && !queued[edge.target] {
                work.push(edge.target);
                queued[edge.target] = true;
            }
        }
    }
    let mut errors = Vec::new();
    let mut unknown_reasons = Vec::new();
    let edge_out: Vec<_> = graph
        .blocks
        .iter()
        .enumerate()
        .map(|(i, block)| transfer_block(block, incoming[i].clone(), fences).1)
        .collect();
    for pending in incoming.iter().chain(edge_out.iter().flatten()) {
        if let Pending::Unknown { reasons, .. } = pending {
            for reason in reasons {
                add_unique(&mut unknown_reasons, *reason, 8);
            }
        }
    }
    for (block_index, block) in graph.blocks.iter().enumerate() {
        let mut pending = incoming[block_index].clone();
        let mut previous_site = None;
        for node in &block.events {
            if node.guard != previous_site && node.guard.is_some_and(|site| fences[site]) {
                pending = Pending::Empty;
            }
            previous_site = node.guard;
            let load = match node.event {
                V4Event::Load(range) | V4Event::Rmw(range, _) => Some(range),
                V4Event::OpaqueCall => Some(V4Range::Unknown),
                _ => None,
            };
            if let Some(load) = load {
                if !pending.permits(load) {
                    let (store_sources, unknown_reasons) = match &pending {
                        Pending::Known(stores) => (
                            stores
                                .iter()
                                .filter(|s| !no_overlap(s.range, load))
                                .map(|s| s.source)
                                .collect(),
                            Vec::new(),
                        ),
                        Pending::Unknown { sources, reasons } => (sources.clone(), reasons.clone()),
                        Pending::Empty => unreachable!(),
                    };
                    errors.push(V4CoverageError {
                        block: block_index,
                        event: node.id,
                        load,
                        path: store_path(graph, block_index, node.id, &store_sources, &edge_out),
                        store_sources,
                        unknown_reasons,
                        guard: node.guard,
                    });
                }
            }
            pending.apply(node.event, node.id);
            if let Pending::Unknown { reasons, .. } = &pending {
                for reason in reasons {
                    add_unique(&mut unknown_reasons, *reason, 8);
                }
            }
        }
    }
    V4Check {
        errors,
        block_in: incoming,
        edge_out,
        unknown_reasons,
    }
}

#[cfg(test)]
fn legacy_plan(graph: &V4Graph) -> Vec<bool> {
    let transfer = |mut dirty: bool, block: &V4Block| {
        let mut states = Vec::new();
        for node in &block.events {
            match node.event {
                V4Event::Load(_) | V4Event::Barrier(_) => dirty = false,
                V4Event::Store(_) | V4Event::Rmw(_, _) | V4Event::OpaqueCall => dirty = true,
                _ => {}
            }
            states.push(dirty);
        }
        (dirty, states)
    };
    let n = graph.blocks.len();
    let mut incoming = vec![false; n];
    incoming[graph.entry] = true;
    loop {
        let mut changed = false;
        for (block_index, block) in graph.blocks.iter().enumerate() {
            let (normal, states) = transfer(incoming[block_index], block);
            for edge in &block.successors {
                let outgoing = edge.cut.map(|cut| states[cut]).unwrap_or(normal);
                if outgoing && !incoming[edge.target] {
                    incoming[edge.target] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut fences = vec![false; graph.sites];
    for (i, block) in graph.blocks.iter().enumerate() {
        let mut dirty = incoming[i];
        for node in &block.events {
            if dirty
                && matches!(
                    node.event,
                    V4Event::Load(_) | V4Event::Rmw(_, _) | V4Event::OpaqueCall
                )
            {
                if let Some(site) = node.guard {
                    fences[site] = true;
                }
            }
            match node.event {
                V4Event::Load(_) | V4Event::Barrier(_) => dirty = false,
                V4Event::Store(_) | V4Event::Rmw(_, _) | V4Event::OpaqueCall => dirty = true,
                _ => {}
            }
        }
    }
    fences
}

fn reverse_postorder(graph: &V4Graph) -> Vec<usize> {
    let mut seen = vec![false; graph.blocks.len()];
    let mut order = Vec::new();
    // An explicit stack avoids recursive traversal on large native functions.
    let mut work = vec![(graph.entry, false)];
    while let Some((block, post)) = work.pop() {
        if post {
            order.push(block);
            continue;
        }
        if seen[block] {
            continue;
        }
        seen[block] = true;
        work.push((block, true));
        for edge in graph.blocks[block].successors.iter().rev() {
            work.push((edge.target, false));
        }
    }
    order.reverse();
    for i in 0..seen.len() {
        if !seen[i] {
            order.push(i);
        }
    }
    order
}

/// Determine natural-loop nesting from DFS backedges. Multiple backedges to
/// one header count as one loop. Irreducible cycles remain subject to the same
/// fully verified local placement rule.
pub fn assign_loop_depths(graph: &mut V4Graph) {
    let n = graph.blocks.len();
    let mut predecessors = vec![Vec::new(); n];
    for (source, block) in graph.blocks.iter().enumerate() {
        for edge in &block.successors {
            predecessors[edge.target].push(source);
        }
    }
    let mut visited = vec![false; n];
    let mut active = vec![false; n];
    let mut backedges = vec![Vec::new(); n];
    let mut work = vec![(graph.entry, false)];
    while let Some((block, post)) = work.pop() {
        if post {
            active[block] = false;
            continue;
        }
        if visited[block] {
            continue;
        }
        visited[block] = true;
        active[block] = true;
        work.push((block, true));
        for edge in graph.blocks[block].successors.iter().rev() {
            if active[edge.target] {
                backedges[edge.target].push(block);
            } else if !visited[edge.target] {
                work.push((edge.target, false));
            }
        }
    }
    for (header, tails) in backedges.into_iter().enumerate() {
        if tails.is_empty() {
            continue;
        }
        let mut members = vec![false; n];
        members[header] = true;
        let mut work = tails;
        while let Some(block) = work.pop() {
            if members[block] {
                continue;
            }
            members[block] = true;
            for pred in &predecessors[block] {
                if !members[*pred] {
                    work.push(*pred);
                }
            }
        }
        for (block, member) in graph.blocks.iter_mut().zip(members) {
            if member {
                block.loop_depth += 1;
            }
        }
    }
}

/// Synthetic fixtures may construct their own boolean legacy approximation.
/// Production callers must provide the original historical replay instead.
#[cfg(test)]
pub fn plan(graph: &V4Graph, policy: V4Policy) -> crate::CodegenResult<V4Plan> {
    plan_with_budget(graph, policy, SOLVE_LIMIT)
}

#[cfg(test)]
fn plan_with_budget(
    graph: &V4Graph,
    policy: V4Policy,
    budget: usize,
) -> crate::CodegenResult<V4Plan> {
    let baseline = V4Baseline {
        fences: legacy_plan(graph),
        fixed_internal_fences: fixed_internal_count(graph),
    };
    plan_baseline_with_budget(graph, policy, &baseline, budget)
}

fn fixed_internal_count(graph: &V4Graph) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.events)
        .filter(|node| node.event == V4Event::Barrier(V4BarrierOrigin::LegacyInternal))
        .count()
}

pub fn plan_with_baseline(
    graph: &V4Graph,
    policy: V4Policy,
    baseline: &V4Baseline,
) -> crate::CodegenResult<V4Plan> {
    plan_baseline_with_budget(graph, policy, baseline, SOLVE_LIMIT)
}

fn plan_baseline_with_budget(
    graph: &V4Graph,
    policy: V4Policy,
    baseline: &V4Baseline,
    budget: usize,
) -> crate::CodegenResult<V4Plan> {
    assert!(budget >= 2);
    if baseline.fences.len() != graph.sites
        || baseline.fixed_internal_fences != fixed_internal_count(graph)
    {
        return Err(crate::CodegenError::Unsupported(format!(
            "Spectre-v4 actual baseline shape/fixed guard mismatch: sites={}, baseline_sites={}, fixed={}, baseline_fixed={}",
            graph.sites,
            baseline.fences.len(),
            fixed_internal_count(graph),
            baseline.fixed_internal_fences
        )));
    }
    let checked_baseline = check(graph, &baseline.fences);
    if let Some(error) = checked_baseline.errors.first() {
        return Err(crate::CodegenError::Unsupported(format!(
            "Spectre-v4 actual legacy baseline failed: {error:?}"
        )));
    }
    let legacy_fences = baseline.fences.iter().filter(|fence| **fence).count();
    let mut count = 1;
    let mut fallback = None;
    let mut moved = 0;
    let mut cfg_moves = Vec::new();
    let mut saved = baseline.fences.clone();
    let mut candidate = vec![false; graph.sites];
    let order = reverse_postorder(graph);
    while count < budget - 1 {
        let checked = check(graph, &candidate);
        count += 1;
        if checked.errors.is_empty() {
            saved = candidate;
            break;
        }
        let error = order
            .iter()
            .find_map(|block| checked.errors.iter().find(|error| error.block == *block))
            .unwrap();
        if let Some(site) = error.guard {
            if candidate[site] {
                return Err(crate::CodegenError::Unsupported(format!(
                    "Spectre-v4 uncovered effect inside indivisible instruction: {error:?}"
                )));
            }
            candidate[site] = true;
        } else {
            return Err(crate::CodegenError::Unsupported(format!(
                "Spectre-v4 uncovered effect without legal fence position: {error:?}"
            )));
        }
    }
    if count >= budget - 1 {
        fallback =
            Some("solve budget exhausted during construction; latest verified plan retained");
    }
    if count < budget - 1 {
        for site in 0..graph.sites {
            if !saved[site] {
                continue;
            }
            if count >= budget - 1 {
                fallback =
                    Some("solve budget exhausted during deletion; latest verified plan retained");
                break;
            }
            let mut trial = saved.clone();
            trial[site] = false;
            let checked = check(graph, &trial);
            count += 1;
            if checked.errors.is_empty() {
                saved = trial;
            }
        }
    }
    // Apply the historical total upper bound before CFG placement, including
    // for RangesCfg: every subsequent increase must come from a recorded move.
    // Fixed internal guards are identical in both plans by the checks above.
    if saved.iter().filter(|f| **f).count() > legacy_fences {
        saved = baseline.fences.clone();
        fallback =
            Some("range plan exceeds actual legacy total; verified actual baseline retained");
    }
    let pre_cfg_sites = saved.clone();
    if policy == V4Policy::RangesCfg && count < budget - 1 {
        let initial_sites = saved.clone();
        let extra_limit = 8.max(saved.iter().filter(|f| **f).count());
        let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
        for (block_index, block) in graph.blocks.iter().enumerate() {
            let mut prefix_pure = true;
            let mut prev = None;
            for node in &block.events {
                if node.guard != prev && prefix_pure {
                    if let Some(site) = node.guard {
                        if initial_sites[site] {
                            candidates.push((block_index, site, block.loop_depth));
                        }
                    }
                }
                prefix_pure &= node.event == V4Event::Pure;
                prev = node.guard;
            }
        }
        candidates.sort_by_key(|(block, site, depth)| (core::cmp::Reverse(*depth), *block, *site));
        for (block_index, site, _) in candidates {
            if !saved[site] {
                continue;
            }
            if count >= budget - 1 {
                fallback = Some(
                    "solve budget exhausted during CFG placement; latest verified plan retained",
                );
                break;
            }
            let current = check(graph, &saved);
            count += 1;
            let mut clean = false;
            let mut dirty_sites = Vec::new();
            let mut legal = true;
            for (pred_index, pred) in graph.blocks.iter().enumerate() {
                for (edge_index, edge) in pred
                    .successors
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.target == block_index)
                {
                    if current.edge_out[pred_index][edge_index].is_empty() {
                        clean = true;
                    } else if pred.successors.len() == 1 && edge.cut.is_none() {
                        if let Some(exit) = pred.exit_site {
                            add_unique(&mut dirty_sites, exit, graph.sites);
                        } else {
                            legal = false;
                        }
                    } else {
                        legal = false;
                    }
                }
            }
            if !legal || !clean || dirty_sites.is_empty() {
                continue;
            }
            let mut trial = saved.clone();
            trial[site] = false;
            for dirty in &dirty_sites {
                trial[*dirty] = true;
            }
            if trial
                .iter()
                .zip(&initial_sites)
                .filter(|(new, old)| **new && !**old)
                .count()
                > extra_limit
            {
                continue;
            }
            if count >= budget - 1 {
                fallback = Some(
                    "solve budget exhausted before CFG verification; latest verified plan retained",
                );
                break;
            }
            let checked = check(graph, &trial);
            count += 1;
            if checked.errors.is_empty() {
                cfg_moves.push(V4CfgMove {
                    removed_site: site,
                    added_sites: dirty_sites
                        .into_iter()
                        .filter(|site| !saved[*site])
                        .collect(),
                });
                saved = trial;
                moved += 1;
            }
        }
    }
    // This solve is reserved even when optimization exhausted its budget.
    let final_check = check(graph, &saved);
    count += 1;
    if let Some(error) = final_check.errors.first() {
        return Err(crate::CodegenError::Unsupported(format!(
            "Spectre-v4 final publication verification failed: {error:?}"
        )));
    }
    let differences = |old: &[bool]| {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        for (site, (new, old)) in saved.iter().zip(old).enumerate() {
            if *new && !*old {
                added.push(site);
            }
            if !*new && *old {
                removed.push(site);
            }
        }
        (added, removed)
    };
    let (cfg_added_sites, cfg_removed_sites) = differences(&pre_cfg_sites);
    let (baseline_added_sites, baseline_removed_sites) = differences(&baseline.fences);
    Ok(V4Plan {
        fences: saved,
        legacy_fences,
        solve_count: count,
        fallback,
        moved,
        cfg_moves,
        cfg_added_sites,
        cfg_removed_sites,
        baseline_added_sites,
        baseline_removed_sites,
        unknown_reasons: final_check.unknown_reasons,
    })
}

/// Decode the final emitted bytes for leaf regression fixtures without data pools.
/// Only barriers, padding NOPs, and direct-branch displacements are normalized;
/// branch predicates, branch counts, targets, and all other operands are retained.
#[cfg(all(test, feature = "disas"))]
pub(crate) fn decoded_nonfence_stream(
    cs: &capstone::Capstone,
    bytes: &[u8],
    barrier: &[u8],
) -> Vec<(alloc::string::String, alloc::string::String)> {
    use alloc::string::ToString;
    let instructions = cs.disasm_all(bytes, 0).unwrap();
    assert_eq!(
        instructions
            .iter()
            .map(|inst| inst.bytes().len())
            .sum::<usize>(),
        bytes.len(),
        "fixture contains undecoded bytes"
    );
    let skip =
        |inst: &capstone::Insn<'_>| inst.bytes() == barrier || inst.mnemonic() == Some("nop");
    let mut positions = Vec::new();
    let mut position = 0;
    for inst in instructions.iter() {
        positions.push((inst.address(), position));
        if !skip(inst) {
            position += 1;
        }
    }
    positions.push((bytes.len() as u64, position));
    instructions
        .iter()
        .filter(|inst| !skip(inst))
        .map(|inst| {
            let mnemonic = inst.mnemonic().unwrap().to_string();
            let mut operands = inst.op_str().unwrap_or("").to_string();
            let detail = cs.insn_detail(inst).unwrap();
            let branch = detail.groups().iter().any(|group| {
                u32::from(group.0) == capstone::InsnGroupType::CS_GRP_JUMP
                    || u32::from(group.0) == capstone::InsnGroupType::CS_GRP_CALL
            });
            if branch {
                if let Some(start) = operands.rfind("0x") {
                    let target = u64::from_str_radix(&operands[start + 2..], 16).unwrap();
                    let target = positions
                        .iter()
                        .find(|(address, _)| *address == target)
                        .unwrap_or_else(|| panic!("unresolved native branch target: {operands}"))
                        .1;
                    operands.replace_range(start.., &format!("@{target}"));
                }
            }
            (mnemonic, operands)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use regalloc2::{PReg, RegClass};

    #[test]
    fn frozen_entry_guard_is_conditional_and_first_read_is_exposed() {
        for steps in [
            vec![SsbbMemStep::Load],
            vec![SsbbMemStep::Call],
            vec![SsbbMemStep::Load, SsbbMemStep::Call],
            vec![SsbbMemStep::Atomic],
        ] {
            let legacy = SsbbSeq {
                steps: steps.into(),
                exn_cut: None,
            };
            let frozen = V4Seq::frozen_legacy(&legacy);
            assert_eq!(frozen.fixed_barriers(), 0, "{legacy:?}: {frozen:?}");
            assert!(matches!(
                frozen.events[0],
                V4Event::Load(_) | V4Event::OpaqueCall | V4Event::Rmw(_, _)
            ));
            let mut events = vec![V4Event::Barrier(V4BarrierOrigin::Upstream)];
            events.extend(frozen.events.iter().copied());
            let clean = graph(events);
            assert!(check(&clean, &vec![false; clean.sites]).errors.is_empty());
            let mut events = vec![V4Event::Store(frame(0, 8))];
            events.extend(frozen.events);
            let dirty = graph(events);
            assert!(
                !check(&dirty, &vec![false; dirty.sites]).errors.is_empty(),
                "dirty input must remain exposed to the checker"
            );
        }
    }

    #[test]
    fn frozen_stack_returns_keep_only_true_internal_guards_and_call_cut() {
        let legacy = SsbbSeq {
            steps: vec![
                SsbbMemStep::Call,
                SsbbMemStep::Load,
                SsbbMemStep::Store,
                SsbbMemStep::Load,
                SsbbMemStep::Store,
            ]
            .into(),
            exn_cut: Some(0),
        };
        let frozen = V4Seq::frozen_legacy(&legacy);
        assert_eq!(frozen.fixed_barriers(), 2);
        assert_eq!(
            frozen.exn_cut,
            Some(0),
            "exception edge must stop immediately after the actual call"
        );
        assert_eq!(frozen.events[0], V4Event::OpaqueCall);
        assert_eq!(
            SsbbSeq::count_barriers(false, &legacy.steps),
            frozen.fixed_barriers()
        );
        let mut events = vec![V4Event::Barrier(V4BarrierOrigin::Upstream)];
        events.extend(frozen.events);
        events.push(V4Event::Load(V4Range::Unknown));
        let g = graph(events);
        assert!(
            !check(&g, &vec![false; g.sites]).errors.is_empty(),
            "a stack-result store or call return remains pending"
        );
    }

    #[test]
    fn frozen_retry_keeps_backedge_guard_without_fixed_entry_or_fake_store() {
        let legacy = SsbbSeq {
            steps: vec![SsbbMemStep::Load, SsbbMemStep::Store, SsbbMemStep::Atomic].into(),
            exn_cut: None,
        };
        let mut frozen = V4Seq::frozen_legacy(&legacy);
        // The backend removes the old state-only retry Store marker.
        frozen
            .events
            .retain(|event| !matches!(event, V4Event::Store(_)));
        assert_eq!(frozen.fixed_barriers(), 1);
        assert_eq!(frozen.events[0], V4Event::Load(V4Range::Unknown));
        assert!(matches!(frozen.events.last(), Some(V4Event::Rmw(_, _))));
        let mut events = vec![V4Event::Barrier(V4BarrierOrigin::Upstream)];
        events.extend(frozen.events);
        events.push(V4Event::Load(V4Range::Unknown));
        let g = graph(events);
        assert!(
            !check(&g, &vec![false; g.sites]).errors.is_empty(),
            "RMW writes must survive the retry expansion"
        );
    }

    #[test]
    fn frame_ranges_total_is_bounded_by_real_legacy_state_replay() {
        let call = SsbbSeq {
            steps: vec![SsbbMemStep::Call].into(),
            exn_cut: None,
        };
        let frozen = V4Seq::frozen_legacy(&call);
        let mut events = vec![
            V4Event::Barrier(V4BarrierOrigin::Upstream),
            V4Event::Load(V4Range::Unknown),
            V4Event::Pure,
        ];
        events.extend(frozen.events.iter().copied());
        events.push(V4Event::Load(V4Range::Unknown));
        let g = graph(events);
        // This is the actual historical emitter trace, not a plan regenerated
        // on a graph after newly added fixed entry guards.
        let original_steps = [
            SsbbMemStep::Clear,
            SsbbMemStep::Load,
            SsbbMemStep::Call,
            SsbbMemStep::Load,
        ];
        let original_count = SsbbSeq::count_barriers(true, &original_steps) as usize;
        assert_eq!(original_count, 1);
        for policy in [V4Policy::Frame, V4Policy::Ranges] {
            let p = plan(&g, policy).unwrap();
            let emitted =
                p.fences.iter().filter(|fence| **fence).count() + frozen.fixed_barriers() as usize;
            assert!(
                emitted <= original_count,
                "{policy:?}: emitted={emitted}, real legacy={original_count}"
            );
        }
    }

    #[test]
    fn non_cfg_replanning_does_not_spread_one_legacy_guard_into_two() {
        let g = V4Graph {
            blocks: vec![
                block(
                    vec![
                        V4Event::Barrier(V4BarrierOrigin::Upstream),
                        V4Event::Store(frame(0, 8)),
                        V4Event::Load(frame(16, 8)),
                        V4Event::Pure,
                    ],
                    &[1, 2],
                    0,
                ),
                block(vec![V4Event::Load(frame(0, 8))], &[], 4),
                block(vec![V4Event::Load(frame(0, 8))], &[], 5),
            ],
            entry: 0,
            sites: 6,
        };
        // Historical emission guards the first load, clearing both successors.
        let real_legacy = [false, false, true, false, false, false];
        assert!(check(&g, &real_legacy).errors.is_empty());
        for policy in [V4Policy::Frame, V4Policy::Ranges] {
            let planned = plan_with_baseline(
                &g,
                policy,
                &V4Baseline {
                    fences: real_legacy.to_vec(),
                    fixed_internal_fences: 0,
                },
            )
            .unwrap();
            assert_eq!(
                planned.fences.iter().filter(|fence| **fence).count(),
                1,
                "{policy:?} cannot replace one real guard with guards on both paths"
            );
            assert_eq!(planned.fences, real_legacy);
            assert!(planned.fallback.is_some(), "cost fallback must be reported");
        }
    }

    #[test]
    fn actual_legacy_replay_separates_entry_and_fixed_internal_guards() {
        let sequence = SsbbSeq {
            steps: vec![
                SsbbMemStep::Call,
                SsbbMemStep::Load,
                SsbbMemStep::Store,
                SsbbMemStep::Load,
            ]
            .into(),
            exn_cut: Some(0),
        };
        let effects = V4Seq::frozen_legacy(&sequence);
        let mut clean = false;
        assert!(!replay_legacy_instruction(&sequence, &effects, &mut clean).unwrap());
        assert!(!clean);
        let mut dirty = true;
        assert!(replay_legacy_instruction(&sequence, &effects, &mut dirty).unwrap());
        assert!(!dirty);
        assert_eq!(effects.fixed_barriers(), 2);
        assert_eq!(effects.exn_cut, Some(0));

        let mut broken = effects.clone();
        broken
            .events
            .insert(0, V4Event::Barrier(V4BarrierOrigin::LegacyInternal));
        assert!(replay_legacy_instruction(&sequence, &broken, &mut false).is_err());
        broken = effects.clone();
        broken
            .events
            .retain(|event| !matches!(event, V4Event::Barrier(_)));
        assert!(replay_legacy_instruction(&sequence, &broken, &mut true).is_err());
    }

    #[test]
    fn actual_baseline_is_verified_and_budget_fallback_uses_exact_guard_sites() {
        let g = graph(vec![
            V4Event::Barrier(V4BarrierOrigin::Upstream),
            V4Event::Store(frame(0, 8)),
            V4Event::Load(frame(16, 8)),
            V4Event::Load(frame(0, 8)),
        ]);
        let baseline = V4Baseline {
            fences: vec![false, false, true, false],
            fixed_internal_fences: 0,
        };
        let p = plan_baseline_with_budget(&g, V4Policy::RangesCfg, &baseline, 2).unwrap();
        assert_eq!(p.fences, baseline.fences);
        assert_eq!(p.solve_count, 2);
        assert!(p.cfg_moves.is_empty());
        assert!(p.baseline_added_sites.is_empty());
        assert!(p.baseline_removed_sites.is_empty());

        let mut broken = baseline.clone();
        broken.fences[2] = false;
        assert!(plan_with_baseline(&g, V4Policy::Ranges, &broken).is_err());
        broken = baseline.clone();
        broken.fixed_internal_fences = 1;
        assert!(plan_with_baseline(&g, V4Policy::Ranges, &broken).is_err());
        broken = baseline.clone();
        broken.fences.pop();
        assert!(plan_with_baseline(&g, V4Policy::Ranges, &broken).is_err());
    }

    fn reg() -> Reg {
        Reg::from_real_reg(PReg::new(0, RegClass::Int))
    }
    fn frame(offset: i64, width: u32) -> V4Range {
        V4Range::Frame {
            frame: 0,
            offset,
            width,
        }
    }
    fn local(offset: i64, width: u32) -> V4Range {
        V4Range::LocalBase {
            block: 0,
            reg: reg(),
            epoch: 0,
            offset,
            width,
        }
    }
    fn block(events: Vec<V4Event>, successors: &[usize], site: usize) -> V4Block {
        V4Block {
            events: events
                .into_iter()
                .enumerate()
                .map(|(i, event)| V4Node {
                    id: V4EventId {
                        block: site as u32,
                        inst: i as u32,
                        sub: 0,
                    },
                    event,
                    guard: Some(site + i),
                })
                .collect(),
            successors: successors
                .iter()
                .map(|target| V4Edge {
                    target: *target,
                    cut: None,
                })
                .collect(),
            exit_site: None,
            loop_depth: 0,
        }
    }
    fn graph(events: Vec<V4Event>) -> V4Graph {
        let sites = events.len();
        V4Graph {
            blocks: vec![block(events, &[], 0)],
            entry: 0,
            sites,
        }
    }

    #[test]
    fn widths_and_low_bits_are_conservative() {
        assert!(no_overlap(frame(0, 8), frame(8, 8)));
        assert!(!no_overlap(frame(0, 8), frame(4, 4)));
        assert!(!no_overlap(frame(i64::MAX, 8), frame(i64::MAX - 4, 8)));
        assert!(!no_overlap(frame(i64::MAX, 1), frame(i64::MIN, 1)));
        assert!(no_overlap(local(0, 8), local(16, 8)));
        assert!(!no_overlap(local(4092, 8), local(0, 4)));
        assert!(!no_overlap(local(0, 8), local(4096, 8)));
        assert!(no_overlap(local(-8, 8), local(0, 8)));
        assert!(!no_overlap(local(0, 0), local(16, 8)));
        assert!(!no_overlap(local(0, 4097), local(16, 8)));
        assert!(!no_overlap(frame(0, 8), local(16, 8)));
    }

    #[test]
    fn fixed_broken_plan_does_not_forget_store_after_disjoint_load() {
        let graph = graph(vec![
            V4Event::Barrier(V4BarrierOrigin::Upstream),
            V4Event::Store(local(0, 8)),
            V4Event::Load(local(16, 8)),
            V4Event::Load(local(0, 8)),
        ]);
        let checked = check(&graph, &vec![false; graph.sites]);
        assert_eq!(checked.errors.len(), 1);
        assert_eq!(checked.errors[0].event.inst, 3);
        assert_eq!(checked.errors[0].store_sources[0].inst, 1);
        let plan = plan(&graph, V4Policy::Ranges).unwrap();
        assert!(!plan.fences[2]);
        assert!(plan.fences[3]);
    }

    #[test]
    fn fixed_broken_plan_retains_all_stores_and_same_address() {
        let graph = graph(vec![
            V4Event::Barrier(V4BarrierOrigin::Upstream),
            V4Event::Store(frame(0, 8)),
            V4Event::Store(frame(16, 8)),
            V4Event::Load(frame(0, 8)),
            V4Event::Load(frame(0, 8)),
        ]);
        assert_eq!(check(&graph, &vec![false; graph.sites]).errors.len(), 2);
        let plan = plan(&graph, V4Policy::Frame).unwrap();
        assert!(plan.fences[3]);
        assert!(!plan.fences[4]);
    }

    #[test]
    fn redefining_base_degrades_pending_instead_of_dropping_it() {
        let graph = graph(vec![
            V4Event::Barrier(V4BarrierOrigin::Upstream),
            V4Event::Store(local(0, 8)),
            V4Event::RegisterDef(reg()),
            V4Event::Load(local(16, 8)),
        ]);
        let checked = check(&graph, &vec![false; graph.sites]);
        assert_eq!(checked.errors.len(), 1);
        assert!(
            checked.errors[0]
                .unknown_reasons
                .contains(&UnknownReason::RegisterRedefined)
        );
    }

    #[test]
    fn local_base_does_not_cross_blocks_or_loop_iterations() {
        let graph = V4Graph {
            blocks: vec![
                block(
                    vec![
                        V4Event::Barrier(V4BarrierOrigin::Upstream),
                        V4Event::Store(local(0, 8)),
                    ],
                    &[1],
                    0,
                ),
                block(vec![V4Event::Load(local(16, 8))], &[1], 2),
            ],
            entry: 0,
            sites: 3,
        };
        let checked = check(&graph, &[false; 3]);
        assert_eq!(checked.errors.len(), 1);
        assert!(
            checked.errors[0]
                .unknown_reasons
                .contains(&UnknownReason::BlockBoundary)
        );
    }

    #[test]
    fn external_entry_is_unknown_and_rmw_call_still_write() {
        let graph = graph(vec![
            V4Event::Load(frame(0, 8)),
            V4Event::Rmw(frame(8, 8), frame(8, 8)),
            V4Event::OpaqueCall,
            V4Event::Load(frame(16, 8)),
        ]);
        assert_eq!(check(&graph, &[false; 4]).errors.len(), 4);
        let plan = plan(&graph, V4Policy::Ranges).unwrap();
        assert!(check(&graph, &plan.fences).errors.is_empty());
        assert!(plan.fences[0]);
        assert!(plan.fences[2]);
        assert!(plan.fences[3]);
    }

    #[test]
    fn range_limit_promotes_and_never_loses_stores() {
        let mut events = vec![V4Event::Barrier(V4BarrierOrigin::Upstream)];
        for i in 0..33 {
            events.push(V4Event::Store(frame(i * 16, 8)));
        }
        events.push(V4Event::Load(frame(4096, 8)));
        let graph = graph(events);
        let checked = check(&graph, &vec![false; graph.sites]);
        assert_eq!(checked.errors.len(), 1);
        assert!(
            checked.errors[0]
                .unknown_reasons
                .contains(&UnknownReason::RangeLimit)
        );
    }

    #[test]
    fn exception_edge_observes_call_cut_before_normal_return_guard() {
        let mut source = block(
            vec![
                V4Event::OpaqueCall,
                V4Event::Barrier(V4BarrierOrigin::LegacyInternal),
                V4Event::Load(frame(8, 8)),
            ],
            &[1, 2],
            0,
        );
        source.successors[0].cut = Some(0);
        let graph = V4Graph {
            blocks: vec![
                source,
                block(vec![V4Event::Load(frame(0, 8))], &[], 3),
                block(vec![V4Event::Load(frame(0, 8))], &[], 4),
            ],
            entry: 0,
            sites: 5,
        };
        let checked = check(&graph, &[true, false, false, false, false]);
        assert_eq!(checked.errors.len(), 1);
        assert_eq!(checked.errors[0].block, 1);
    }

    #[test]
    fn prologue_is_separate_external_node_and_internal_backedge_stays_clean() {
        let graph = V4Graph {
            blocks: vec![
                block(vec![V4Event::Load(frame(0, 8))], &[1], 0),
                block(vec![V4Event::Load(frame(8, 8))], &[1], 1),
            ],
            entry: 0,
            sites: 2,
        };
        let plan = plan(&graph, V4Policy::Ranges).unwrap();
        assert!(plan.fences[0]);
        assert!(!plan.fences[1]);
    }

    #[test]
    fn cfg_moves_join_fence_to_only_dirty_single_successor() {
        let mut dirty = block(vec![V4Event::Store(frame(0, 8)), V4Event::Pure], &[3], 1);
        dirty.exit_site = Some(2);
        let graph = V4Graph {
            blocks: vec![
                block(
                    vec![V4Event::Barrier(V4BarrierOrigin::Upstream)],
                    &[1, 2],
                    0,
                ),
                dirty,
                block(vec![V4Event::Pure], &[3], 3),
                block(vec![V4Event::Load(frame(0, 8))], &[], 4),
            ],
            entry: 0,
            sites: 5,
        };
        let plan = plan(&graph, V4Policy::RangesCfg).unwrap();
        assert!(plan.fences[2]);
        assert!(!plan.fences[4]);
        assert_eq!(plan.moved, 1);
        assert_eq!(plan.cfg_moves.len(), 1);
        assert_eq!(plan.cfg_moves[0].removed_site, 4);
        assert_eq!(plan.cfg_moves[0].added_sites, [2]);
        assert_eq!(plan.cfg_added_sites, [2]);
        assert_eq!(plan.cfg_removed_sites, [4]);
        assert_eq!(plan.baseline_added_sites, [2]);
        assert_eq!(plan.baseline_removed_sites, [4]);
    }

    #[test]
    fn cfg_does_not_split_critical_edges() {
        let mut dirty = block(vec![V4Event::Store(frame(0, 8)), V4Event::Pure], &[3, 2], 1);
        dirty.exit_site = Some(2);
        let graph = V4Graph {
            blocks: vec![
                block(
                    vec![V4Event::Barrier(V4BarrierOrigin::Upstream)],
                    &[1, 2],
                    0,
                ),
                dirty,
                block(vec![V4Event::Pure], &[3], 3),
                block(vec![V4Event::Load(frame(0, 8))], &[], 4),
            ],
            entry: 0,
            sites: 5,
        };
        let plan = plan(&graph, V4Policy::RangesCfg).unwrap();
        assert!(plan.fences[4]);
        assert_eq!(plan.moved, 0);
    }

    #[test]
    fn budget_exhaustion_retains_verified_baseline_and_final_solve() {
        let graph = graph(vec![
            V4Event::Load(frame(0, 8)),
            V4Event::Store(frame(0, 8)),
            V4Event::Load(frame(0, 8)),
        ]);
        let plan = plan_with_budget(&graph, V4Policy::RangesCfg, 2).unwrap();
        assert_eq!(plan.solve_count, 2);
        assert!(plan.fallback.is_some());
        assert!(check(&graph, &plan.fences).errors.is_empty());
        assert_eq!(plan.fences, [true, false, true]);
    }

    #[test]
    fn readonly_loop_hoists_only_to_its_dirty_external_predecessor() {
        let mut outside = block(vec![V4Event::Store(frame(0, 8)), V4Event::Pure], &[1], 0);
        outside.exit_site = Some(1);
        let mut header = block(vec![V4Event::Load(frame(0, 8)), V4Event::Pure], &[1], 2);
        header.exit_site = Some(3);
        header.loop_depth = 1;
        let graph = V4Graph {
            blocks: vec![outside, header],
            entry: 0,
            sites: 4,
        };
        let plan = plan(&graph, V4Policy::RangesCfg).unwrap();
        assert!(plan.fences[1]);
        assert!(!plan.fences[2]);
        assert!(check(&graph, &plan.fences).errors.is_empty());
        let mut broken = plan.fences;
        broken[1] = false;
        assert!(!check(&graph, &broken).errors.is_empty());
    }

    #[test]
    fn write_loop_rejects_broken_outside_only_plan() {
        let mut outside = block(vec![V4Event::Store(frame(0, 8)), V4Event::Pure], &[1], 0);
        outside.exit_site = Some(1);
        let header = block(
            vec![V4Event::Load(frame(0, 8)), V4Event::Store(frame(0, 8))],
            &[1],
            2,
        );
        let graph = V4Graph {
            blocks: vec![outside, header],
            entry: 0,
            sites: 4,
        };
        let checked = check(&graph, &[false, true, false, false]);
        assert_eq!(checked.errors.len(), 1);
        assert_eq!(checked.errors[0].block, 1);
        let plan = plan(&graph, V4Policy::RangesCfg).unwrap();
        assert!(plan.fences[2]);
    }
}
