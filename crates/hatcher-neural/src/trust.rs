//! Layer 4: the dynamic trust graph, and the plasticity that follows it.
//!
//! Two matrices evolve here, and keeping them separate is the whole point:
//!
//! * `T` — trust. `T_ij(t+1) = T_ij(t) + λ S_ij − μ E_ij`
//! * `W` — connection strength. `W_ij(t+1) = W_ij(t) + φ T_ij − ψ·latency`
//!
//! Trust is a belief updated by outcomes. Weight is what the mesh actually spends on
//! that belief, discounted by what the link costs. An agent can be trusted and still
//! be routed around because it is slow; an agent that is fast and wrong loses both.
//!
//! The result is a self-organizing communication graph: unreliable agents drift into
//! isolation, reliable ones accumulate inbound trust and become hubs.

use std::collections::HashMap;

use hatcher_core::{MeshCoefficients, MeshEdge, TrustMatrixView};
use ndarray::Array2;

use crate::equations::{plasticity_update, trust_update};

/// Trust extended to an unproven agent: neither credulous nor hostile.
pub const INITIAL_TRUST: f64 = 0.5;
/// Starting connection strength — weak enough that emergence must be earned.
pub const INITIAL_WEIGHT: f64 = 0.10;
/// Default normalized link latency before real timings are recorded.
pub const INITIAL_LATENCY: f64 = 0.20;
/// An agent whose mean inbound weight is below this is effectively unrouted.
pub const ISOLATION_WEIGHT: f64 = 0.05;
/// Inbound trust below which the mesh has actively lost faith in an agent, as opposed
/// to simply never having tried it.
pub const ISOLATION_TRUST: f64 = 0.35;
/// Inbound trust this many times the cohort mean marks a hub.
pub const HUB_FACTOR: f64 = 1.25;

/// What one settlement pass did to the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct TrustSettlement {
    /// Directed pairs whose trust or weight moved.
    pub updated_pairs: usize,
    /// `C` for the `Ω` update: trust-weighted successful handoffs over all handoffs.
    pub collaboration_efficiency: f64,
    /// Interactions considered in this pass.
    pub interactions: f64,
}

/// The trust and plasticity matrices over a cohort of agents.
#[derive(Debug, Clone)]
pub struct TrustGraph {
    ids: Vec<String>,
    index: HashMap<String, usize>,
    trust: Array2<f64>,
    weight: Array2<f64>,
    latency: Array2<f64>,
    successes: Array2<f64>,
    failures: Array2<f64>,
    pending_success: Array2<f64>,
    pending_failure: Array2<f64>,
}

impl TrustGraph {
    /// Build a fully-connected graph at baseline trust over the given agent ids.
    pub fn new<I, S>(ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let ids: Vec<String> = ids.into_iter().map(Into::into).collect();
        let n = ids.len();
        let index = ids
            .iter()
            .enumerate()
            .map(|(position, id)| (id.clone(), position))
            .collect();

        let mut graph = Self {
            ids,
            index,
            trust: Array2::from_elem((n, n), INITIAL_TRUST),
            weight: Array2::from_elem((n, n), INITIAL_WEIGHT),
            latency: Array2::from_elem((n, n), INITIAL_LATENCY),
            successes: Array2::zeros((n, n)),
            failures: Array2::zeros((n, n)),
            pending_success: Array2::zeros((n, n)),
            pending_failure: Array2::zeros((n, n)),
        };
        graph.clear_diagonal();
        graph
    }

    fn clear_diagonal(&mut self) {
        for i in 0..self.ids.len() {
            self.trust[[i, i]] = 0.0;
            self.weight[[i, i]] = 0.0;
            self.latency[[i, i]] = 0.0;
        }
    }

    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn index_of(&self, id: &str) -> Option<usize> {
        self.index.get(id).copied()
    }

    /// Add an agent, preserving every existing value. Returns its index.
    pub fn add_agent(&mut self, id: impl Into<String>) -> usize {
        let id = id.into();
        if let Some(existing) = self.index_of(&id) {
            return existing;
        }

        let old = self.ids.len();
        let new = old + 1;
        self.ids.push(id.clone());
        self.index.insert(id, old);

        self.trust = grow(&self.trust, new, INITIAL_TRUST);
        self.weight = grow(&self.weight, new, INITIAL_WEIGHT);
        self.latency = grow(&self.latency, new, INITIAL_LATENCY);
        self.successes = grow(&self.successes, new, 0.0);
        self.failures = grow(&self.failures, new, 0.0);
        self.pending_success = grow(&self.pending_success, new, 0.0);
        self.pending_failure = grow(&self.pending_failure, new, 0.0);
        self.clear_diagonal();
        old
    }

    /// Set the normalized latency of a directed link.
    pub fn set_latency(&mut self, from: &str, to: &str, latency: f64) -> bool {
        match (self.index_of(from), self.index_of(to)) {
            (Some(i), Some(j)) if i != j => {
                self.latency[[i, j]] = latency.clamp(0.0, 1.0);
                true
            }
            _ => false,
        }
    }

    /// Set latency in both directions, as measured between two agents.
    pub fn set_link_latency(&mut self, a: &str, b: &str, latency: f64) -> bool {
        self.set_latency(a, b, latency) && self.set_latency(b, a, latency)
    }

    /// Record one collaboration outcome from `from` to `to`.
    ///
    /// This only accumulates evidence; nothing moves until [`Self::settle`] runs, so
    /// a whole pipeline pass is applied as a single coherent update.
    pub fn record(&mut self, from: &str, to: &str, success: bool) -> bool {
        match (self.index_of(from), self.index_of(to)) {
            (Some(i), Some(j)) if i != j => {
                if success {
                    self.pending_success[[i, j]] += 1.0;
                    self.successes[[i, j]] += 1.0;
                } else {
                    self.pending_failure[[i, j]] += 1.0;
                    self.failures[[i, j]] += 1.0;
                }
                true
            }
            _ => false,
        }
    }

    /// Apply the trust and plasticity equations to all accumulated evidence.
    ///
    /// `gate` scales how strongly this pass is allowed to move the graph — a sandbox
    /// run teaches the mesh less than a production one. See
    /// [`hatcher_core::ExecutionMode::plasticity_gate`].
    pub fn settle(&mut self, coefficients: &MeshCoefficients, gate: f64) -> TrustSettlement {
        let gate = gate.clamp(0.0, 1.0);
        let n = self.ids.len();
        let mut updated_pairs = 0usize;
        let mut trusted_successes = 0.0;
        let mut interactions = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let successes = self.pending_success[[i, j]] * gate;
                let failures = self.pending_failure[[i, j]] * gate;
                let observed = self.pending_success[[i, j]] + self.pending_failure[[i, j]];

                if observed > 0.0 {
                    // Weight this pair's contribution to C by how much the mesh
                    // trusts it: a success across a distrusted link says less about
                    // the mesh's collaboration quality than one across a trusted link.
                    trusted_successes +=
                        self.pending_success[[i, j]] * (0.5 + 0.5 * self.trust[[i, j]]);
                    interactions += observed;
                }

                let before_trust = self.trust[[i, j]];
                let before_weight = self.weight[[i, j]];

                self.trust[[i, j]] = trust_update(before_trust, successes, failures, coefficients);

                // Plasticity is asymmetric on purpose. A link that carried traffic this
                // pass gets the full rule — it strengthens with trust and pays for its
                // latency. A link that carried nothing pays the latency cost anyway.
                //
                // Without that asymmetry the graph saturates: baseline trust on an
                // unproven pair is high enough that `φT` outruns `ψ·latency`, so every
                // pair would thicken whether or not it was ever used, and the mesh would
                // converge on fully-connected instead of organizing itself.
                self.weight[[i, j]] = if observed > 0.0 {
                    plasticity_update(
                        before_weight,
                        self.trust[[i, j]],
                        self.latency[[i, j]],
                        coefficients,
                    )
                } else {
                    plasticity_update(before_weight, 0.0, self.latency[[i, j]], coefficients)
                };

                if (self.trust[[i, j]] - before_trust).abs() > f64::EPSILON
                    || (self.weight[[i, j]] - before_weight).abs() > f64::EPSILON
                {
                    updated_pairs += 1;
                }
            }
        }

        self.pending_success.fill(0.0);
        self.pending_failure.fill(0.0);

        TrustSettlement {
            updated_pairs,
            collaboration_efficiency: if interactions > 0.0 {
                (trusted_successes / interactions).clamp(0.0, 1.0)
            } else {
                0.0
            },
            interactions,
        }
    }

    pub fn trust_between(&self, from: &str, to: &str) -> f64 {
        self.lookup(&self.trust, from, to)
    }

    pub fn weight_between(&self, from: &str, to: &str) -> f64 {
        self.lookup(&self.weight, from, to)
    }

    pub fn latency_between(&self, from: &str, to: &str) -> f64 {
        self.lookup(&self.latency, from, to)
    }

    fn lookup(&self, matrix: &Array2<f64>, from: &str, to: &str) -> f64 {
        match (self.index_of(from), self.index_of(to)) {
            (Some(i), Some(j)) => matrix[[i, j]],
            _ => 0.0,
        }
    }

    /// Mean trust the cohort extends *into* this agent — its reputation.
    pub fn inbound_trust(&self, id: &str) -> f64 {
        self.column_mean(&self.trust, id)
    }

    /// Mean trust this agent extends *out* to the cohort — its credulity.
    pub fn outbound_trust(&self, id: &str) -> f64 {
        self.row_mean(&self.trust, id)
    }

    /// Mean connection strength routed into this agent.
    pub fn inbound_weight(&self, id: &str) -> f64 {
        self.column_mean(&self.weight, id)
    }

    fn column_mean(&self, matrix: &Array2<f64>, id: &str) -> f64 {
        let Some(j) = self.index_of(id) else {
            return 0.0;
        };
        let n = self.ids.len();
        if n < 2 {
            return 0.0;
        }
        (0..n)
            .filter(|i| *i != j)
            .map(|i| matrix[[i, j]])
            .sum::<f64>()
            / (n - 1) as f64
    }

    fn row_mean(&self, matrix: &Array2<f64>, id: &str) -> f64 {
        let Some(i) = self.index_of(id) else {
            return 0.0;
        };
        let n = self.ids.len();
        if n < 2 {
            return 0.0;
        }
        (0..n)
            .filter(|j| *j != i)
            .map(|j| matrix[[i, j]])
            .sum::<f64>()
            / (n - 1) as f64
    }

    /// Live connections into or out of this agent.
    pub fn degree(&self, id: &str) -> usize {
        let Some(position) = self.index_of(id) else {
            return 0;
        };
        let n = self.ids.len();
        (0..n)
            .filter(|other| {
                *other != position
                    && (self.weight[[position, *other]] > ISOLATION_WEIGHT
                        || self.weight[[*other, position]] > ISOLATION_WEIGHT)
            })
            .count()
    }

    /// Mean trust across every directed pair — a single number for mesh cohesion.
    pub fn mean_trust(&self) -> f64 {
        let n = self.ids.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sum += self.trust[[i, j]];
                }
            }
        }
        sum / (n * (n - 1)) as f64
    }

    /// Agents that have become hubs: inbound trust well above the cohort mean.
    pub fn hubs(&self) -> Vec<String> {
        let mean = self.mean_trust();
        if mean <= 0.0 {
            return Vec::new();
        }
        self.ids
            .iter()
            .filter(|id| {
                let inbound = self.inbound_trust(id);
                inbound >= mean * HUB_FACTOR && inbound > 0.5
            })
            .cloned()
            .collect()
    }

    /// Agents the mesh has actively stopped routing to.
    ///
    /// Both conditions are required, and the second is the important one: a thin
    /// inbound weight alone only means "idle", which is the normal state of a fresh
    /// agent nobody has tried yet. Isolation means the cohort has *judged* the agent
    /// and stopped trusting it. Conflating the two would starve every new agent, since
    /// the router skips isolated candidates.
    pub fn isolated(&self) -> Vec<String> {
        self.ids
            .iter()
            .filter(|id| {
                self.inbound_weight(id) < ISOLATION_WEIGHT
                    && self.inbound_trust(id) < ISOLATION_TRUST
            })
            .cloned()
            .collect()
    }

    /// Agents no traffic currently flows to, whether or not they are distrusted.
    pub fn idle(&self) -> Vec<String> {
        self.ids
            .iter()
            .filter(|id| self.inbound_weight(id) < ISOLATION_WEIGHT)
            .cloned()
            .collect()
    }

    /// The `k` partners this agent should talk to, best first.
    ///
    /// Ranked by `W_ij · T_ij`: the mesh prefers partners it both trusts and can
    /// afford to reach.
    pub fn top_partners(&self, id: &str, k: usize) -> Vec<(String, f64)> {
        let Some(i) = self.index_of(id) else {
            return Vec::new();
        };
        let mut partners: Vec<(String, f64)> = (0..self.ids.len())
            .filter(|j| *j != i)
            .map(|j| {
                (
                    self.ids[j].clone(),
                    self.weight[[i, j]] * self.trust[[i, j]],
                )
            })
            .collect();
        partners.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        partners.truncate(k);
        partners
    }

    /// `W`, for the mesh-intelligence and message-passing computations.
    pub fn weight_matrix(&self) -> &Array2<f64> {
        &self.weight
    }

    /// `T`, for message gating.
    pub fn trust_matrix(&self) -> &Array2<f64> {
        &self.trust
    }

    /// Materialize the graph as serializable edges.
    pub fn edges(&self) -> Vec<MeshEdge> {
        let n = self.ids.len();
        let mut edges = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                edges.push(MeshEdge {
                    from: self.ids[i].clone(),
                    to: self.ids[j].clone(),
                    weight: self.weight[[i, j]],
                    trust: self.trust[[i, j]],
                    latency: self.latency[[i, j]],
                    signal: self.weight[[i, j]] * self.trust[[i, j]],
                    energy: 0.0,
                    successes: self.successes[[i, j]],
                    failures: self.failures[[i, j]],
                });
            }
        }
        edges
    }

    /// Read model for the trust-matrix endpoint.
    pub fn to_view(&self) -> TrustMatrixView {
        TrustMatrixView {
            ids: self.ids.clone(),
            trust: self
                .trust
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
            weight: self
                .weight
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
        }
    }
}

/// Copy a square matrix into a larger one, filling new cells with `fill`.
fn grow(source: &Array2<f64>, size: usize, fill: f64) -> Array2<f64> {
    let mut grown = Array2::from_elem((size, size), fill);
    let rows = source.nrows().min(size);
    let cols = source.ncols().min(size);
    for i in 0..rows {
        for j in 0..cols {
            grown[[i, j]] = source[[i, j]];
        }
    }
    grown
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cohort() -> TrustGraph {
        TrustGraph::new(["planner", "coder", "critic"])
    }

    #[test]
    fn new_graph_starts_at_baseline_with_no_self_trust() {
        let graph = cohort();
        assert_eq!(graph.len(), 3);
        assert_eq!(graph.trust_between("planner", "coder"), INITIAL_TRUST);
        assert_eq!(
            graph.trust_between("planner", "planner"),
            0.0,
            "no self-trust"
        );
        assert_eq!(
            graph.trust_between("planner", "ghost"),
            0.0,
            "unknown agents are unknown"
        );
    }

    #[test]
    fn reliable_partners_gain_trust_and_weight() {
        let mut graph = cohort();
        for _ in 0..6 {
            graph.record("planner", "coder", true);
            graph.settle(&MeshCoefficients::default(), 1.0);
        }
        assert!(graph.trust_between("planner", "coder") > INITIAL_TRUST);
        assert!(graph.weight_between("planner", "coder") > INITIAL_WEIGHT);
    }

    #[test]
    fn failing_partners_are_routed_around() {
        let mut graph = cohort();
        let coefficients = MeshCoefficients::default();
        for _ in 0..30 {
            graph.record("planner", "critic", false);
            graph.record("coder", "critic", false);
            graph.settle(&coefficients, 1.0);
        }
        assert_eq!(graph.trust_between("planner", "critic"), 0.0);
        assert!(graph.inbound_weight("critic") < ISOLATION_WEIGHT);
        assert!(graph.isolated().contains(&"critic".to_string()));
        assert!(!graph.isolated().contains(&"coder".to_string()));
    }

    #[test]
    fn consistent_success_produces_a_hub() {
        let mut graph = cohort();
        let coefficients = MeshCoefficients::default();
        for _ in 0..10 {
            graph.record("planner", "coder", true);
            graph.record("critic", "coder", true);
            graph.record("coder", "critic", false);
            graph.settle(&coefficients, 1.0);
        }
        assert!(
            graph.hubs().contains(&"coder".to_string()),
            "hubs: {:?}",
            graph.hubs()
        );
        assert!(graph.inbound_trust("coder") > graph.inbound_trust("critic"));
    }

    #[test]
    fn an_untried_agent_is_idle_not_isolated() {
        let mut graph = cohort();
        let coefficients = MeshCoefficients::default();
        // Nobody ever works with the critic; its links wither from disuse.
        for _ in 0..40 {
            graph.record("planner", "coder", true);
            graph.settle(&coefficients, 1.0);
        }

        assert!(
            graph.idle().contains(&"critic".to_string()),
            "no traffic flows to it"
        );
        assert!(
            !graph.isolated().contains(&"critic".to_string()),
            "but the mesh has not judged it, so it must stay eligible for work"
        );
        assert!(graph.inbound_trust("critic") >= ISOLATION_TRUST);
    }

    #[test]
    fn latency_suppresses_weight_at_equal_trust() {
        let mut graph = TrustGraph::new(["a", "b", "c"]);
        graph.set_link_latency("a", "b", 0.0);
        graph.set_link_latency("a", "c", 1.0);
        let coefficients = MeshCoefficients::default();
        for _ in 0..5 {
            graph.record("a", "b", true);
            graph.record("a", "c", true);
            graph.settle(&coefficients, 1.0);
        }
        assert!(
            graph.weight_between("a", "b") > graph.weight_between("a", "c"),
            "the expensive link must stay thinner even though trust is equal"
        );
        assert!((graph.trust_between("a", "b") - graph.trust_between("a", "c")).abs() < 1e-12);
    }

    #[test]
    fn settlement_reports_collaboration_efficiency() {
        let mut graph = cohort();
        let coefficients = MeshCoefficients::default();

        let idle = graph.settle(&coefficients, 1.0);
        assert_eq!(idle.interactions, 0.0);
        assert_eq!(idle.collaboration_efficiency, 0.0);

        graph.record("planner", "coder", true);
        graph.record("planner", "critic", true);
        graph.record("coder", "critic", false);
        let settlement = graph.settle(&coefficients, 1.0);
        assert_eq!(settlement.interactions, 3.0);
        assert!(
            settlement.collaboration_efficiency > 0.0 && settlement.collaboration_efficiency < 1.0
        );
        assert!(settlement.updated_pairs > 0);
    }

    #[test]
    fn the_gate_throttles_how_much_a_sandbox_run_teaches() {
        let coefficients = MeshCoefficients::default();
        let mut sandbox = cohort();
        let mut production = cohort();

        sandbox.record("planner", "coder", true);
        sandbox.settle(&coefficients, 0.25);
        production.record("planner", "coder", true);
        production.settle(&coefficients, 1.0);

        assert!(
            sandbox.trust_between("planner", "coder")
                < production.trust_between("planner", "coder")
        );
    }

    #[test]
    fn pending_evidence_is_consumed_exactly_once() {
        let mut graph = cohort();
        let coefficients = MeshCoefficients::default();
        graph.record("planner", "coder", true);
        graph.settle(&coefficients, 1.0);
        let after_first = graph.trust_between("planner", "coder");
        graph.settle(&coefficients, 1.0);
        let after_second = graph.trust_between("planner", "coder");
        assert!(
            after_second <= after_first,
            "settling twice must not re-apply the same success"
        );
    }

    #[test]
    fn adding_an_agent_preserves_learned_trust() {
        let mut graph = cohort();
        let coefficients = MeshCoefficients::default();
        for _ in 0..4 {
            graph.record("planner", "coder", true);
            graph.settle(&coefficients, 1.0);
        }
        let learned = graph.trust_between("planner", "coder");

        let position = graph.add_agent("verifier");
        assert_eq!(position, 3);
        assert_eq!(graph.len(), 4);
        assert_eq!(graph.trust_between("planner", "coder"), learned);
        assert_eq!(graph.trust_between("planner", "verifier"), INITIAL_TRUST);
        assert_eq!(graph.add_agent("verifier"), 3, "adding twice is a no-op");
    }

    #[test]
    fn top_partners_ranks_by_trust_times_weight() {
        let mut graph = cohort();
        let coefficients = MeshCoefficients::default();
        for _ in 0..8 {
            graph.record("planner", "coder", true);
            graph.record("planner", "critic", false);
            graph.settle(&coefficients, 1.0);
        }
        let partners = graph.top_partners("planner", 2);
        assert_eq!(partners.len(), 2);
        assert_eq!(partners[0].0, "coder");
        assert!(partners[0].1 > partners[1].1);
        assert!(graph.top_partners("ghost", 2).is_empty());
    }

    #[test]
    fn edges_and_view_expose_the_full_matrices() {
        let graph = cohort();
        let edges = graph.edges();
        assert_eq!(edges.len(), 6, "3 agents give 6 directed edges");
        assert!(edges.iter().all(|edge| edge.from != edge.to));

        let view = graph.to_view();
        assert_eq!(view.ids.len(), 3);
        assert_eq!(view.trust.len(), 3);
        assert_eq!(view.weight[0].len(), 3);
    }

    #[test]
    fn degree_counts_live_links_only() {
        let mut graph = cohort();
        let coefficients = MeshCoefficients::default();
        for _ in 0..30 {
            graph.record("planner", "critic", false);
            graph.record("coder", "critic", false);
            graph.record("critic", "planner", false);
            graph.record("critic", "coder", false);
            graph.settle(&coefficients, 1.0);
        }
        assert_eq!(
            graph.degree("critic"),
            0,
            "a fully distrusted agent has no live links"
        );
        assert_eq!(graph.degree("ghost"), 0);
    }

    #[test]
    fn recording_unknown_agents_is_rejected_not_silently_dropped() {
        let mut graph = cohort();
        assert!(!graph.record("planner", "ghost", true));
        assert!(
            !graph.record("planner", "planner", true),
            "self-collaboration is meaningless"
        );
        assert!(graph.record("planner", "coder", true));
        assert!(!graph.set_latency("planner", "ghost", 0.5));
    }
}
