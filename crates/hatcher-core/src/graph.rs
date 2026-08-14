//! Layers 2–4 of the mesh: the graph itself — nodes, weighted/trusted edges, and
//! the aggregate intelligence that emerges from them.

use serde::{Deserialize, Serialize};

use crate::agent::AgentNode;
use crate::global::GlobalState;

/// A directed edge from one agent to another.
///
/// The edge carries two distinct quantities that are easy to conflate:
///
/// * `trust` is `T_ij`, the belief that `j` will do good work for `i`. It moves via
///   `T_ij(t+1) = T_ij(t) + λ S_ij − μ E_ij`.
/// * `weight` is `W_ij`, the connection strength actually used for message passing.
///   It follows trust but pays for latency: `W_ij(t+1) = W_ij + φ T_ij − ψ·latency`.
///
/// Trust is what the mesh believes; weight is what the mesh can afford.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeshEdge {
    pub from: String,
    pub to: String,
    /// `W_ij` — connection strength, in `[0, 1]`.
    pub weight: f64,
    /// `T_ij` — trust, in `[0, 1]`.
    pub trust: f64,
    /// Normalized communication cost on this link.
    pub latency: f64,
    /// Magnitude of the last message passed along this edge.
    pub signal: f64,
    /// Shared memory mass across the two endpoints — how much context transfers.
    pub energy: f64,
    /// `S_ij` — cumulative successful collaborations.
    pub successes: f64,
    /// `E_ij` — cumulative execution failures.
    pub failures: f64,
}

impl MeshEdge {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            weight: 0.1,
            trust: 0.5,
            latency: 0.2,
            signal: 0.0,
            energy: 0.0,
            successes: 0.0,
            failures: 0.0,
        }
    }

    /// Whether this link is effectively pruned — trust and weight both collapsed.
    pub fn is_dormant(&self) -> bool {
        self.weight < 0.02 && self.trust < 0.15
    }
}

/// The decomposition of mesh intelligence `A`.
///
/// `A = Σ_i A_i + γ Σ_{i≠j} A_i A_j W_ij`
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct MeshIntelligence {
    /// `Σ A_i` — raw capability, what the cohort is worth disconnected.
    pub raw: f64,
    /// `γ Σ_{i≠j} A_i A_j W_ij` — collective intelligence from connection.
    pub emergent: f64,
    /// `A` — the sum of the two.
    pub total: f64,
}

impl MeshIntelligence {
    /// Share of total intelligence attributable to connection rather than headcount.
    ///
    /// This is the `E` term fed into the `Ω` update: it is exactly "how much of what
    /// this mesh can do could not be done by its agents in isolation".
    pub fn emergence_ratio(&self) -> f64 {
        if self.total.abs() < 1e-12 {
            return 0.0;
        }
        (self.emergent / self.total).clamp(0.0, 1.0)
    }
}

/// The full mesh state at one instant.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeshState {
    pub nodes: Vec<AgentNode>,
    pub edges: Vec<MeshEdge>,
    pub global: GlobalState,
}

impl MeshState {
    pub fn new(nodes: Vec<AgentNode>) -> Self {
        Self {
            nodes,
            edges: Vec::new(),
            global: GlobalState::default(),
        }
    }

    pub fn index_of(&self, id: &str) -> Option<usize> {
        self.nodes.iter().position(|node| node.id == id)
    }

    pub fn node(&self, id: &str) -> Option<&AgentNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    pub fn node_mut(&mut self, id: &str) -> Option<&mut AgentNode> {
        self.nodes.iter_mut().find(|node| node.id == id)
    }

    pub fn edge(&self, from: &str, to: &str) -> Option<&MeshEdge> {
        self.edges
            .iter()
            .find(|edge| edge.from == from && edge.to == to)
    }

    /// Live edges, strongest connection first.
    pub fn active_edges(&self) -> Vec<&MeshEdge> {
        let mut edges: Vec<&MeshEdge> = self
            .edges
            .iter()
            .filter(|edge| !edge.is_dormant())
            .collect();
        edges.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        edges
    }
}

/// One evolution step of the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStepResult {
    pub step_index: usize,
    pub nodes: Vec<AgentNode>,
    pub edges: Vec<MeshEdge>,
    /// Post-message-passing activation per node, aligned with `nodes`.
    pub activations: Vec<f64>,
    pub intelligence: MeshIntelligence,
    /// `Ω` after this step.
    pub omega: f64,
    /// Scheduling pressure for the step, from the priority equation.
    pub pressure: f64,
    pub digest: String,
}

/// A multi-step rehearsal of the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshSimulation {
    pub request_id: String,
    pub steps: Vec<MeshStepResult>,
    pub final_omega: f64,
    pub final_pressure: f64,
    pub final_intelligence: MeshIntelligence,
    pub final_digest: String,
}

impl MeshSimulation {
    /// Change in `Ω` across the whole rehearsal.
    pub fn omega_drift(&self) -> f64 {
        match (self.steps.first(), self.steps.last()) {
            (Some(first), Some(last)) => last.omega - first.omega,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentRole;

    #[test]
    fn emergence_ratio_isolates_the_pairwise_term() {
        let intelligence = MeshIntelligence {
            raw: 3.0,
            emergent: 1.0,
            total: 4.0,
        };
        assert!((intelligence.emergence_ratio() - 0.25).abs() < 1e-12);
        assert_eq!(MeshIntelligence::default().emergence_ratio(), 0.0);
    }

    #[test]
    fn dormant_edges_are_excluded_from_active_set() {
        let mut state = MeshState::new(vec![
            AgentNode::new("a", "a", AgentRole::Planner),
            AgentNode::new("b", "b", AgentRole::Coder),
        ]);
        let mut live = MeshEdge::new("a", "b");
        live.weight = 0.7;
        let mut dead = MeshEdge::new("b", "a");
        dead.weight = 0.001;
        dead.trust = 0.01;
        state.edges = vec![dead, live];

        let active = state.active_edges();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].to, "b");
        assert!(state.edge("b", "a").unwrap().is_dormant());
    }

    #[test]
    fn state_lookups_resolve_by_id() {
        let state = MeshState::new(vec![AgentNode::new(
            "planner-1",
            "planner",
            AgentRole::Planner,
        )]);
        assert_eq!(state.index_of("planner-1"), Some(0));
        assert!(state.node("planner-1").is_some());
        assert!(state.node("missing").is_none());
    }
}
