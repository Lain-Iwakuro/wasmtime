#![allow(missing_docs)]
//! Final-machine-code Spectre-v4 events and independently checked fence plans.

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
    /// Its emitter must start legacy dirty=true even when the outer plan is clean.
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

    /// Preserve all legacy expansion guards as fixed events, including an
    /// unconditional entry guard when its first operation reads memory.
    pub fn frozen_legacy(seq: &SsbbSeq) -> Self {
        let mut result = Self::neutral();
        result.legacy_internal = true;
        let mut dirty = true;
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
    /// Legal position immediately before a pure unconditional branch.
    pub exit_site: Option<usize>,
    pub loop_depth: usize,
}

#[derive(Clone, Debug)]
pub struct V4Graph {
    pub blocks: Vec<V4Block>,
    pub entry: usize,
    pub sites: usize,
}

#[derive(Clone, Debug)]
pub struct V4Plan {
    pub fences: Vec<bool>,
    pub legacy_fences: usize,
    pub solve_count: usize,
    pub fallback: Option<&'static str>,
    pub moved: usize,
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

pub fn plan(graph: &V4Graph, policy: V4Policy) -> crate::CodegenResult<V4Plan> {
    plan_with_budget(graph, policy, SOLVE_LIMIT)
}

fn plan_with_budget(
    graph: &V4Graph,
    policy: V4Policy,
    budget: usize,
) -> crate::CodegenResult<V4Plan> {
    assert!(budget >= 2);
    let legacy = legacy_plan(graph);
    let baseline = check(graph, &legacy);
    if let Some(error) = baseline.errors.first() {
        return Err(crate::CodegenError::Unsupported(format!(
            "Spectre-v4 conservative baseline failed: {error:?}"
        )));
    }
    let legacy_fences = legacy.iter().filter(|fence| **fence).count();
    let mut count = 1;
    let mut fallback = None;
    let mut moved = 0;
    let mut saved = legacy;
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
            for dirty in dirty_sites {
                trial[dirty] = true;
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
    Ok(V4Plan {
        fences: saved,
        legacy_fences,
        solve_count: count,
        fallback,
        moved,
        unknown_reasons: final_check.unknown_reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use regalloc2::{PReg, RegClass};

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
