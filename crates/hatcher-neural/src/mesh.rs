//! The mesh engine: graph structure, message passing, and the stateful runtime that
//! ties all ten equations to real observations.
//!
//! ## Graph structure
//!
//! Nodes are [`AgentNode`]s. Edges live in the [`TrustGraph`] as two matrices, `T`
//! and `W`. There is no separate adjacency list: an edge exists exactly to the degree
//! that `W_ij > 0`, which is why an untrusted agent is not "removed" from the mesh so
//! much as forgotten by it.
//!
//! ## Message passing
//!
//! One round is
//!
//! ```text
//! m_i = Σ_{j≠i} W_ij · T_ij · A_j · h_j
//! h_i ← squash(h_i + m_i)
//! ```
//!
//! which is a graph neural network layer with three deliberate choices: messages are
//! gated by trust as well as weight, they are scaled by the sender's capability `A_j`
//! (a confident but incapable agent cannot shout down the mesh), and the update is
//! residual, so an isolated node keeps its own signal instead of decaying to zero.

use hatcher_core::{
    canonical_digest, fold_digests, AgentNode, AgentRole, AgentSummary, CapabilityVector, DecisionHead,
    ExecutionMode, GlobalState, MemoryGraph, MeshAnalytics, MeshCoefficients, MeshConfigView, MeshEdge,
    MeshGraphView, MeshIntelligence, MeshOverview, MeshSimulation, MeshState, MeshStepResult, ModelSpec,
    OmegaDelta, OmegaLedger, OmegaSample, PriorityScore, ResourceProfile, TaskSpec, TrustMatrixView,
};

use crate::equations::{
    self, global_intelligence, mesh_intelligence, priority, squash, urgency_gain, OMEGA_CEILING,
};
use crate::inference::{resolve_or_native, BackendKind, Decision, InferenceError, SharedBackend};
use crate::trust::TrustGraph;

/// How many rounds of message passing one mesh step performs.
///
/// Two rounds let information travel two hops — far enough for a planner to feel a
/// verifier's state through a coder — without the over-smoothing that makes every
/// node in a deep GNN look alike.
pub const MESSAGE_ROUNDS: usize = 2;

/// The stateful mesh runtime.
#[derive(Debug, Clone)]
pub struct NeuralMesh {
    pub coefficients: MeshCoefficients,
    pub nodes: Vec<AgentNode>,
    pub trust: TrustGraph,
    pub memory: MemoryGraph,
    pub global: GlobalState,
    pub ledger: OmegaLedger,
    backend: SharedBackend,
    backend_kind: BackendKind,
    /// Set when the requested backend could not be loaded and the mesh fell back.
    degraded: Option<String>,
    /// Monotonic task counter, used for deterministic ids.
    pub sequence: u64,
}

impl Default for NeuralMesh {
    fn default() -> Self {
        Self::with_cohort(default_cohort())
    }
}

impl NeuralMesh {
    /// Build a mesh with the default pipeline cohort and a native head of the given shape.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let spec = ModelSpec::new("hamns-native", "2.0.0", input_dim, hidden_dim, output_dim);
        let (backend, _) = resolve_or_native(&BackendKind::Native, spec);
        let mut mesh = Self::with_cohort(default_cohort());
        mesh.backend = backend;
        mesh
    }

    /// Build a mesh over an explicit cohort.
    pub fn with_cohort(nodes: Vec<AgentNode>) -> Self {
        let (backend, _) = resolve_or_native(&BackendKind::Native, ModelSpec::native_default());
        let trust = TrustGraph::new(nodes.iter().map(|node| node.id.clone()));

        let mut mesh = Self {
            coefficients: MeshCoefficients::default(),
            nodes,
            trust,
            memory: MemoryGraph::default(),
            global: GlobalState::default(),
            ledger: OmegaLedger::default(),
            backend,
            backend_kind: BackendKind::Native,
            degraded: None,
            sequence: 0,
        };
        mesh.sync_link_latencies();
        mesh
    }

    pub fn with_coefficients(mut self, coefficients: MeshCoefficients) -> Self {
        self.coefficients = coefficients;
        self
    }

    /// Bind a specific inference backend, falling back to the native head on failure.
    ///
    /// The fallback is recorded in [`Self::degraded`] rather than hidden, so an
    /// operator can see that the mesh is not running the model they asked for.
    pub fn with_backend(mut self, kind: BackendKind, spec: ModelSpec) -> Self {
        let (backend, error) = resolve_or_native(&kind, spec);
        self.degraded = error.map(|error| error.to_string());
        self.backend_kind = kind;
        self.backend = backend;
        self
    }

    /// Bind a backend, surfacing the load failure instead of degrading.
    pub fn try_with_backend(mut self, kind: BackendKind, spec: ModelSpec) -> Result<Self, InferenceError> {
        self.backend = kind.resolve(spec)?;
        self.backend_kind = kind;
        self.degraded = None;
        Ok(self)
    }

    /// Load an ONNX policy as the decision head.
    pub fn try_with_onnx(self, model_path: impl Into<String>) -> Result<Self, InferenceError> {
        let spec = ModelSpec::native_default();
        self.try_with_backend(
            BackendKind::Onnx {
                model_path: model_path.into(),
            },
            spec,
        )
    }

    pub fn backend(&self) -> &SharedBackend {
        &self.backend
    }

    pub fn backend_kind(&self) -> &BackendKind {
        &self.backend_kind
    }

    /// A message explaining why the mesh is not running its requested backend, if so.
    pub fn degraded(&self) -> Option<&str> {
        self.degraded.as_deref()
    }

    /// The decision head, as the integration contract describes it.
    pub fn head(&self) -> DecisionHead {
        let spec = self.backend.spec();
        DecisionHead::new(
            self.backend.name(),
            &spec.name,
            &spec.version,
            spec.input_dim,
            spec.output_dim,
        )
        .with_degraded(self.degraded.clone())
    }

    // -----------------------------------------------------------------------
    // Cohort management
    // -----------------------------------------------------------------------

    /// Add an agent to the mesh, wiring it into the trust graph at baseline trust.
    pub fn add_agent(&mut self, node: AgentNode) {
        if self.node(&node.id).is_some() {
            return;
        }
        self.trust.add_agent(node.id.clone());
        self.nodes.push(node);
        self.sync_link_latencies();
    }

    /// Set every link's latency from the endpoints' own latency profiles.
    pub fn sync_link_latencies(&mut self) {
        let profiles: Vec<(String, f64)> = self
            .nodes
            .iter()
            .map(|node| (node.id.clone(), node.resources.latency))
            .collect();

        for (from, from_latency) in &profiles {
            for (to, to_latency) in &profiles {
                if from == to {
                    continue;
                }
                self.trust.set_latency(from, to, (from_latency + to_latency) / 2.0);
            }
        }
    }

    pub fn node(&self, id: &str) -> Option<&AgentNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    pub fn node_mut(&mut self, id: &str) -> Option<&mut AgentNode> {
        self.nodes.iter_mut().find(|node| node.id == id)
    }

    /// The first agent holding a role, preferring higher capability.
    pub fn agent_for_role(&self, role: AgentRole) -> Option<&AgentNode> {
        self.nodes
            .iter()
            .filter(|node| node.role == role)
            .max_by(|a, b| {
                a.influence()
                    .partial_cmp(&b.influence())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    // -----------------------------------------------------------------------
    // Intelligence
    // -----------------------------------------------------------------------

    /// `A_i` for every node, in cohort order.
    pub fn capabilities(&self) -> Vec<f64> {
        self.nodes.iter().map(|node| node.influence()).collect()
    }

    /// `A = Σ A_i + γ Σ_{i≠j} A_i A_j W_ij`
    pub fn intelligence(&self) -> MeshIntelligence {
        mesh_intelligence(
            &self.capabilities(),
            self.trust.weight_matrix(),
            self.coefficients.gamma,
        )
    }

    /// The collaborations currently producing the most collective intelligence.
    pub fn top_collaborations(&self, limit: usize) -> Vec<(String, String, f64)> {
        equations::top_emergent_pairs(
            &self.capabilities(),
            self.trust.weight_matrix(),
            self.coefficients.gamma,
            limit,
        )
        .into_iter()
        .filter_map(|(i, j, contribution)| {
            Some((self.nodes.get(i)?.id.clone(), self.nodes.get(j)?.id.clone(), contribution))
        })
        .collect()
    }

    pub fn mean_confidence(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.nodes.iter().map(|node| node.confidence).sum::<f64>() / self.nodes.len() as f64
    }

    pub fn mean_memory(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.nodes.iter().map(|node| node.capability.memory).sum::<f64>() / self.nodes.len() as f64
    }

    /// Mesh confidence about a specific domain: confidence weighted by mastery.
    pub fn domain_confidence(&self, domain: &str) -> f64 {
        let mut weight = 0.0;
        let mut total = 0.0;
        for node in &self.nodes {
            let mastery = node.mastery(domain);
            weight += mastery;
            total += mastery * node.confidence;
        }
        if weight <= 0.0 {
            return self.mean_confidence();
        }
        (total / weight).clamp(0.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // Message passing
    // -----------------------------------------------------------------------

    /// Spread a task's feature vector across the cohort as initial activations.
    ///
    /// Features are dealt round-robin so a short vector still touches every node,
    /// then biased by the node's own capability: a capable agent starts louder.
    pub fn seed_activations(&self, features: &[f64]) -> Vec<f64> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let feature = if features.is_empty() {
                    0.0
                } else {
                    features[index % features.len()]
                };
                squash(feature + node.influence())
            })
            .collect()
    }

    /// Run [`MESSAGE_ROUNDS`] rounds of trust-gated message passing.
    pub fn message_pass(&self, seeds: &[f64]) -> Vec<f64> {
        self.message_pass_rounds(seeds, MESSAGE_ROUNDS)
    }

    /// Run an explicit number of message-passing rounds.
    pub fn message_pass_rounds(&self, seeds: &[f64], rounds: usize) -> Vec<f64> {
        let n = self.nodes.len();
        let mut activations: Vec<f64> = (0..n).map(|index| seeds.get(index).copied().unwrap_or(0.0)).collect();
        if n < 2 {
            return activations;
        }

        let capabilities = self.capabilities();
        let weights = self.trust.weight_matrix();
        let trust = self.trust.trust_matrix();

        for _ in 0..rounds {
            let mut next = activations.clone();
            for i in 0..n {
                let mut message = 0.0;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    message += weights[[i, j]] * trust[[i, j]] * capabilities[j] * activations[j];
                }
                next[i] = squash(activations[i] + message);
            }
            activations = next;
        }

        activations
    }

    // -----------------------------------------------------------------------
    // The decision head
    // -----------------------------------------------------------------------

    /// The canonical model input: four mesh statistics, then four task statistics.
    ///
    /// | Slot | Meaning |
    /// |---|---|
    /// | 0 | `Ω` normalized by [`OMEGA_CEILING`] |
    /// | 1 | emergence ratio — share of `A` that comes from connection |
    /// | 2 | mean trust across the cohort |
    /// | 3 | mean calibrated confidence |
    /// | 4 | priority `P`, squashed into `(0, 1)` |
    /// | 5 | task uncertainty `U` |
    /// | 6 | task budget `B` |
    /// | 7 | task implementation cost `I` |
    ///
    /// Mesh statistics come first so a narrower model still sees the mesh's own state.
    /// Raw task features follow slot 7 for models wide enough to use them.
    pub fn model_input(&self, task: &TaskSpec, priority: &PriorityScore) -> Vec<f64> {
        let intelligence = self.intelligence();
        let mut input = vec![
            (self.global.omega / OMEGA_CEILING).clamp(0.0, 1.0),
            intelligence.emergence_ratio(),
            self.trust.mean_trust(),
            self.mean_confidence(),
            squash(priority.value),
            task.uncertainty,
            task.budget,
            task.implementation_cost,
        ];
        input.extend(task.features.iter().copied().filter(|value| value.is_finite()));
        input
    }

    /// Ask the decision head what to do about a task.
    pub fn decide(&self, task: &TaskSpec, priority: &PriorityScore) -> Result<Decision, InferenceError> {
        self.backend.decide(&self.model_input(task, priority))
    }

    /// Priority for a task, using the mesh's confidence in that task's domain.
    pub fn priority_for(&self, task: &TaskSpec) -> PriorityScore {
        priority(
            task.uncertainty,
            task.budget,
            task.implementation_cost,
            self.domain_confidence(&task.domain),
            urgency_gain(task.urgency),
            &self.coefficients,
        )
    }

    // -----------------------------------------------------------------------
    // Evolution
    // -----------------------------------------------------------------------

    /// One rehearsal step: pass messages, let context follow the mesh signal, and
    /// move `Ω`.
    ///
    /// This is deliberately weaker than a pipeline run. Nothing was executed, so `F`
    /// is zero and `L` is only the context the cohort gained by talking to itself. A
    /// mesh cannot rehearse its way to a high `Ω`.
    pub fn step(&mut self, task: &TaskSpec, step_index: usize) -> MeshStepResult {
        let seeds = self.seed_activations(&task.features);
        let activations = self.message_pass(&seeds);

        let mut context_gain = 0.0;
        for (node, activation) in self.nodes.iter_mut().zip(activations.iter()) {
            let before = node.capability.context;
            // Context awareness tracks how much mesh signal actually reaches a node.
            node.capability.context = (before * 0.85 + activation * 0.15).clamp(0.0, 1.0);
            context_gain += (node.capability.context - before).max(0.0);
        }
        let cohort = self.nodes.len().max(1) as f64;

        let intelligence = self.intelligence();
        let priority = self.priority_for(task);

        let delta = OmegaDelta {
            learning: (context_gain / cohort).clamp(0.0, 1.0),
            emergence: intelligence.emergence_ratio(),
            collaboration: self.trust.mean_trust(),
            failure: 0.0,
            // A rehearsal that the mesh is not confident about is drift, not learning.
            drift: (1.0 - self.mean_confidence()).clamp(0.0, 1.0) * 0.5,
        };

        self.apply_omega(delta, intelligence);

        let edges = self.edges();
        let digest = self.digest();

        MeshStepResult {
            step_index,
            nodes: self.nodes.clone(),
            edges,
            activations,
            intelligence,
            omega: self.global.omega,
            pressure: priority.value,
            digest,
        }
    }

    /// Apply an `Ω` update and record it in the ledger.
    pub fn apply_omega(&mut self, delta: OmegaDelta, intelligence: MeshIntelligence) {
        self.global.omega = global_intelligence(self.global.omega, &delta, &self.coefficients);
        self.global.last = delta;
        self.global.epoch += 1;
        self.global.aggregate_intelligence = intelligence.total;

        self.ledger.record(OmegaSample {
            epoch: self.global.epoch,
            omega: self.global.omega,
            delta,
            aggregate_intelligence: intelligence.total,
            regime_gain: delta.gain(self.coefficients.alpha),
            regime_penalty: delta.penalty(self.coefficients.beta),
        });
    }

    /// Rehearse a task for several steps without mutating the live mesh.
    pub fn simulate(&self, task: &TaskSpec, steps: usize) -> MeshSimulation {
        let mut working = self.clone();
        let mut history = Vec::new();

        for step_index in 1..=steps.max(1) {
            history.push(working.step(task, step_index));
        }

        let last = history.last().expect("at least one step always runs");
        MeshSimulation {
            request_id: task.id.clone(),
            steps: history.clone(),
            final_omega: last.omega,
            final_pressure: last.pressure,
            final_intelligence: last.intelligence,
            final_digest: last.digest.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Serialization and read models
    // -----------------------------------------------------------------------

    /// Edges with their memory energy filled in from the endpoint nodes.
    pub fn edges(&self) -> Vec<MeshEdge> {
        let mut edges = self.trust.edges();
        for edge in &mut edges {
            let from = self.node(&edge.from).map(|node| node.capability.memory).unwrap_or(0.0);
            let to = self.node(&edge.to).map(|node| node.capability.memory).unwrap_or(0.0);
            edge.energy = (from + to) / 2.0;
        }
        edges
    }

    /// The full serializable mesh state.
    pub fn state(&self) -> MeshState {
        MeshState {
            nodes: self.nodes.clone(),
            edges: self.edges(),
            global: self.global,
        }
    }

    /// A single commitment over the whole mesh: every node, the edges, `Ω`, and the head.
    ///
    /// The decision head is in here because it is not decoration — it is the thing that
    /// turns mesh state into the control action a caller acts on. Two meshes with
    /// identical nodes and identical trust but different policies return different
    /// answers, so a commitment that ignored the head would attest to state while saying
    /// nothing about what that state was used to decide.
    pub fn digest(&self) -> String {
        let mut digests: Vec<String> = self
            .nodes
            .iter()
            .map(|node| canonical_digest(node).unwrap_or_default())
            .collect();
        digests.push(canonical_digest(&self.edges()).unwrap_or_default());
        digests.push(canonical_digest(&self.global).unwrap_or_default());
        digests.push(canonical_digest(&self.head().commitment()).unwrap_or_default());
        fold_digests(digests)
    }

    /// Summarize one node. `hubs` and `isolated` are passed in rather than recomputed
    /// per node — both are cohort-wide scans, and recomputing them inside the loop
    /// would make a roster render cubic in the number of agents.
    fn summarize(&self, node: &AgentNode, hubs: &[String], isolated: &[String]) -> AgentSummary {
        let hub = hubs.contains(&node.id);
        let isolated = isolated.contains(&node.id);

        AgentSummary {
            id: node.id.clone(),
            label: node.label.clone(),
            role: node.role.as_str().to_string(),
            capability: node.influence(),
            bottleneck: node.capability.bottleneck().0.to_string(),
            confidence: node.confidence,
            efficiency: node.resource_efficiency(),
            inbound_trust: self.trust.inbound_trust(&node.id),
            outbound_trust: self.trust.outbound_trust(&node.id),
            degree: self.trust.degree(&node.id),
            attempts: node.telemetry.attempts,
            successes: node.telemetry.successes,
            failures: node.telemetry.failures,
            isolated,
            hub,
        }
    }

    /// Roster of every agent, strongest first.
    pub fn agent_summaries(&self) -> Vec<AgentSummary> {
        let hubs = self.trust.hubs();
        let isolated = self.trust.isolated();
        let mut summaries: Vec<AgentSummary> = self
            .nodes
            .iter()
            .map(|node| self.summarize(node, &hubs, &isolated))
            .collect();
        summaries.sort_by(|a, b| b.capability.partial_cmp(&a.capability).unwrap_or(std::cmp::Ordering::Equal));
        summaries
    }

    /// Overview-tab read model.
    pub fn overview(&self) -> MeshOverview {
        let intelligence = self.intelligence();
        MeshOverview {
            omega: self.global.omega,
            regime: self.global.regime(),
            epoch: self.global.epoch,
            intelligence,
            emergence_ratio: intelligence.emergence_ratio(),
            agent_count: self.nodes.len(),
            active_edge_count: self.edges().iter().filter(|edge| !edge.is_dormant()).count(),
            agents: self.agent_summaries(),
            hubs: self.trust.hubs(),
            isolated: self.trust.isolated(),
        }
    }

    /// Analytics-tab read model.
    pub fn analytics(&self) -> MeshAnalytics {
        let attempts: u64 = self.nodes.iter().map(|node| node.telemetry.attempts).sum();
        let successes: u64 = self.nodes.iter().map(|node| node.telemetry.successes).sum();

        MeshAnalytics {
            omega: self.global.omega,
            omega_slope: self.ledger.slope(16),
            series: self.ledger.samples().to_vec(),
            intelligence: self.intelligence(),
            mean_confidence: self.mean_confidence(),
            mean_trust: self.trust.mean_trust(),
            total_tasks: attempts,
            success_rate: if attempts == 0 {
                0.0
            } else {
                successes as f64 / attempts as f64
            },
            agents: self.agent_summaries(),
        }
    }

    /// Graph-tab read model.
    pub fn graph_view(&self) -> MeshGraphView {
        MeshGraphView {
            nodes: self.agent_summaries(),
            edges: self.edges(),
            intelligence: self.intelligence(),
            omega: self.global.omega,
        }
    }

    /// Trust-matrix read model.
    pub fn trust_view(&self) -> TrustMatrixView {
        self.trust.to_view()
    }

    /// Config-tab read model.
    pub fn config_view(&self, execution_mode: ExecutionMode) -> MeshConfigView {
        MeshConfigView {
            coefficients: self.coefficients,
            agent_count: self.nodes.len(),
            execution_mode,
        }
    }
}

/// The default five-station cohort: planner, researcher, coder, critic, verifier.
///
/// Two things are calibrated here rather than picked arbitrarily.
///
/// The vectors are **differentiated** — a researcher with wide context and a coder with
/// deep specialization route differently — and the resource profiles differ so
/// [`AgentNode::resource_efficiency`] has something to choose on.
///
/// They are also **competent**: fitness lands around `0.75`, which puts each staffed
/// stage near a 75% success rate on routine work. That matters more than it looks,
/// because performance `P` is measured from outcomes and feeds straight back into `A_i`.
/// A cohort tuned to coin-flip on easy work does not stay at a coin flip — failures
/// depress `P`, which depresses capability, which produces more failures, and `Ω`
/// spirals to the floor. The default cohort has to sit comfortably on the right side of
/// that feedback loop; a cohort that does not is exactly what `Ω` is there to reveal.
pub fn default_cohort() -> Vec<AgentNode> {
    vec![
        AgentNode::new("planner-01", "planner", AgentRole::Planner)
            .with_capability(CapabilityVector::new(0.88, 0.78, 0.82, 0.88, 0.75))
            .with_resources(ResourceProfile::new(0.30, 0.20))
            .with_confidence(0.60)
            .with_expertise("general", 0.75)
            .with_expertise("planning", 0.90),
        AgentNode::new("researcher-01", "researcher", AgentRole::Researcher)
            .with_capability(CapabilityVector::new(0.85, 0.72, 0.80, 0.90, 0.82))
            .with_resources(ResourceProfile::new(0.45, 0.40))
            .with_confidence(0.55)
            .with_expertise("research", 0.88)
            .with_expertise("general", 0.72),
        AgentNode::new("coder-01", "coder", AgentRole::Coder)
            .with_capability(CapabilityVector::new(0.87, 0.88, 0.82, 0.78, 0.72))
            .with_resources(ResourceProfile::new(0.55, 0.35))
            .with_confidence(0.62)
            .with_expertise("rust", 0.90)
            .with_expertise("general", 0.70),
        AgentNode::new("critic-01", "critic", AgentRole::Critic)
            .with_capability(CapabilityVector::new(0.88, 0.80, 0.85, 0.80, 0.70))
            .with_resources(ResourceProfile::new(0.25, 0.15))
            .with_confidence(0.65)
            .with_expertise("review", 0.88)
            .with_expertise("general", 0.74),
        AgentNode::new("verifier-01", "verifier", AgentRole::Verifier)
            .with_capability(CapabilityVector::new(0.82, 0.85, 0.90, 0.78, 0.68))
            .with_resources(ResourceProfile::new(0.20, 0.10))
            .with_confidence(0.70)
            .with_expertise("verification", 0.90)
            .with_expertise("general", 0.72),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::PipelineStage;

    fn task() -> TaskSpec {
        TaskSpec::new("task-1", "wire the router", "rust")
            .with_features(vec![0.3, 0.7, 0.2, 0.9])
            .parsed_from_features()
    }

    #[test]
    fn default_mesh_staffs_every_pipeline_role() {
        let mesh = NeuralMesh::default();
        assert_eq!(mesh.nodes.len(), 5);
        for stage in PipelineStage::ALL {
            if let Some(role) = stage.preferred_role() {
                assert!(mesh.agent_for_role(role).is_some(), "no agent for {role:?}");
            }
        }
    }

    #[test]
    fn connection_creates_intelligence_that_isolation_does_not() {
        // Same cohort, same capability, two histories: one that collaborated and one
        // that sat idle while its links decayed.
        let mut collaborating = NeuralMesh::default();
        let mut idle = NeuralMesh::default();
        let coefficients = collaborating.coefficients;

        for _ in 0..30 {
            collaborating.trust.record("planner-01", "coder-01", true);
            collaborating.trust.record("coder-01", "critic-01", true);
            collaborating.trust.settle(&coefficients, 1.0);
            idle.trust.settle(&coefficients, 1.0);
        }

        let connected = collaborating.intelligence();
        let alone = idle.intelligence();

        assert!((connected.raw - alone.raw).abs() < 1e-12, "raw capability is identical");
        assert_eq!(alone.emergent, 0.0, "an idle cohort's links decay to nothing");
        assert!(connected.emergent > 0.0, "collaboration is where collective intelligence lives");
        assert!(connected.total > alone.total);
        assert!(connected.emergence_ratio() > alone.emergence_ratio());
    }

    #[test]
    fn message_passing_moves_signal_between_connected_agents() {
        let mut mesh = NeuralMesh::default();
        for _ in 0..15 {
            mesh.trust.record("planner-01", "coder-01", true);
            mesh.trust.settle(&mesh.coefficients.clone(), 1.0);
        }

        let mut seeds = vec![0.0; mesh.nodes.len()];
        seeds[0] = 1.0;
        let activations = mesh.message_pass(&seeds);

        let coder = mesh.nodes.iter().position(|node| node.id == "coder-01").unwrap();
        assert!(activations[coder] > 0.0, "a trusted neighbour must receive signal");
        assert_eq!(activations.len(), mesh.nodes.len());
        assert!(activations.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn a_single_node_mesh_passes_no_messages() {
        let mesh = NeuralMesh::with_cohort(vec![AgentNode::new("solo", "solo", AgentRole::Executor)]);
        assert_eq!(mesh.message_pass(&[0.4]), vec![0.4]);
        assert_eq!(mesh.intelligence().emergent, 0.0);
    }

    #[test]
    fn model_input_follows_the_canonical_layout() {
        let mesh = NeuralMesh::default();
        let task = task();
        let priority = mesh.priority_for(&task);
        let input = mesh.model_input(&task, &priority);

        assert!(input.len() >= 8);
        assert!((input[0] - mesh.global.omega / OMEGA_CEILING).abs() < 1e-12);
        assert!((input[3] - mesh.mean_confidence()).abs() < 1e-12);
        assert!((input[5] - task.uncertainty).abs() < 1e-12);
        assert!(input.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn decisions_come_back_as_control_actions() {
        let mesh = NeuralMesh::default();
        let task = task();
        let priority = mesh.priority_for(&task);
        let decision = mesh.decide(&task, &priority).unwrap();
        assert!(Decision::ACTION_ORDER.contains(&decision.action));
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn domain_confidence_weights_by_mastery() {
        let mesh = NeuralMesh::default();
        let rust = mesh.domain_confidence("rust");
        let unknown = mesh.domain_confidence("astrophysics");
        assert!(rust > 0.0);
        assert!((unknown - mesh.mean_confidence()).abs() < 0.2, "unknown domains fall back to the mean");
    }

    #[test]
    fn rehearsal_steps_move_omega_and_stay_bounded() {
        let mut mesh = NeuralMesh::default();
        let task = task();
        let before = mesh.global.omega;

        let step = mesh.step(&task, 1);
        assert_eq!(step.step_index, 1);
        assert_eq!(step.nodes.len(), 5);
        assert!(!step.edges.is_empty());
        assert!(!step.digest.is_empty());
        assert!(step.omega >= 0.0 && step.omega <= OMEGA_CEILING);
        assert_ne!(step.omega, before, "a step must actually move the mesh");
        assert_eq!(mesh.global.epoch, 1);
        assert_eq!(mesh.ledger.len(), 1);
    }

    #[test]
    fn simulation_does_not_mutate_the_live_mesh() {
        let mesh = NeuralMesh::default();
        let simulation = mesh.simulate(&task(), 4);

        assert_eq!(simulation.steps.len(), 4);
        assert_eq!(simulation.steps[0].step_index, 1);
        assert_eq!(mesh.global.epoch, 0, "rehearsal is side-effect free");
        assert!(mesh.ledger.is_empty());
        assert!(!simulation.final_digest.is_empty());
    }

    #[test]
    fn rehearsal_alone_cannot_make_the_mesh_smart() {
        let mut mesh = NeuralMesh::default();
        let task = task();
        for step in 1..=40 {
            mesh.step(&task, step);
        }
        assert!(
            mesh.global.omega < OMEGA_CEILING,
            "talking to itself is not the same as doing work"
        );
    }

    #[test]
    fn adding_an_agent_extends_the_graph_and_latency_map() {
        let mut mesh = NeuralMesh::default();
        let guardian = AgentNode::new("guardian-01", "guardian", AgentRole::Guardian)
            .with_resources(ResourceProfile::new(0.1, 0.9));
        mesh.add_agent(guardian);

        assert_eq!(mesh.nodes.len(), 6);
        assert_eq!(mesh.trust.len(), 6);
        assert!(mesh.trust.latency_between("guardian-01", "planner-01") > 0.0);

        mesh.add_agent(AgentNode::new("guardian-01", "dupe", AgentRole::Guardian));
        assert_eq!(mesh.nodes.len(), 6, "ids are unique");
    }

    #[test]
    fn digest_changes_when_state_changes() {
        let mut mesh = NeuralMesh::default();
        let before = mesh.digest();
        assert_eq!(before, mesh.digest(), "the digest is stable for fixed state");
        mesh.step(&task(), 1);
        assert_ne!(before, mesh.digest());
    }

    #[test]
    fn read_models_expose_the_whole_mesh() {
        let mut mesh = NeuralMesh::default();
        mesh.step(&task(), 1);

        let overview = mesh.overview();
        assert_eq!(overview.agent_count, 5);
        assert_eq!(overview.agents.len(), 5);
        assert!(overview.agents[0].capability >= overview.agents[1].capability);

        let analytics = mesh.analytics();
        assert_eq!(analytics.series.len(), 1);
        assert_eq!(analytics.total_tasks, 0);

        let graph = mesh.graph_view();
        assert_eq!(graph.edges.len(), 20, "5 agents give 20 directed edges");

        let trust = mesh.trust_view();
        assert_eq!(trust.ids.len(), 5);

        let config = mesh.config_view(ExecutionMode::Controlled);
        assert_eq!(config.agent_count, 5);
        assert_eq!(config.execution_mode, ExecutionMode::Controlled);
    }

    #[test]
    fn top_collaborations_name_the_pairs_that_matter() {
        let mut mesh = NeuralMesh::default();
        for _ in 0..15 {
            mesh.trust.record("planner-01", "coder-01", true);
            mesh.trust.settle(&mesh.coefficients.clone(), 1.0);
        }
        let pairs = mesh.top_collaborations(3);
        assert!(!pairs.is_empty());
        assert!(pairs
            .iter()
            .any(|(from, to, _)| from == "planner-01" && to == "coder-01"));
    }

    #[test]
    fn an_unavailable_backend_degrades_visibly() {
        let mesh = NeuralMesh::default().with_backend(
            BackendKind::Onnx {
                model_path: "models/missing.onnx".into(),
            },
            ModelSpec::native_default(),
        );

        assert_eq!(mesh.backend().name(), "native", "the mesh keeps running");
        assert!(mesh.degraded().is_some(), "and says why it is not running onnx");
    }
}
