//! Turning a live [`NeuralMesh`] into a renderable observation.
//!
//! The renderer never reaches into the mesh directly. Everything it draws comes
//! through [`Observation`], which is a normalized, already-clamped snapshot —
//! so a panel cannot accidentally divide by a zero cohort or plot a raw `Ω` on a
//! `[0, 1]` axis, and the whole view layer is testable without a mesh.

use hatcher_core::{
    DecisionHead, MeshEdge, MeshIntelligence, OutcomeProvenance, PipelineTrace, RuntimeCalibration,
    CONTRACT_VERSION,
};
use hatcher_neural::{router, NeuralMesh};
use hatcher_playground::BenchmarkReport;

use crate::tornado::{Tornado, VortexNode};

/// One agent, as the observatory sees it.
#[derive(Debug, Clone)]
pub struct AgentView {
    pub id: String,
    pub label: String,
    pub role: String,
    /// `A_i = I·S·P·C·M`, raw.
    pub capability: f64,
    /// `A_i` rescaled against the strongest agent in the cohort.
    pub capability_norm: f64,
    pub confidence: f64,
    /// Mean inbound trust, in `[0, 1]`.
    pub trust: f64,
    /// Live activation after message passing, rescaled to `[0, 1]`.
    pub activation: f64,
    pub attempts: u64,
    pub successes: u64,
    pub failures: u64,
    pub last_action: Option<String>,
    /// `R_i = P_i / (Energy_i + Latency_i)`.
    pub efficiency: f64,
    /// What the mesh expects a stage by this agent to take, in milliseconds.
    pub expected_latency_ms: f64,
    /// What it expects a stage to cost, in the caller's cost units.
    pub expected_cost: f64,
    /// How many real outcome reports have landed on this agent. Zero means every
    /// number above it is still a declaration rather than a measurement.
    pub observations: u64,
}

impl AgentView {
    /// Success rate over attempts, or `None` for an unproven agent.
    pub fn success_rate(&self) -> Option<f64> {
        if self.attempts == 0 {
            return None;
        }
        Some(self.successes as f64 / self.attempts as f64)
    }

    /// Whether this agent's cost and latency are measured or merely declared.
    pub fn is_measured(&self) -> bool {
        self.observations > 0
    }
}

/// A normalized snapshot of the whole mesh at one instant.
#[derive(Debug, Clone)]
pub struct Observation {
    pub agents: Vec<AgentView>,
    pub edges: Vec<MeshEdge>,
    pub intelligence: MeshIntelligence,
    pub omega: f64,
    pub epoch: u64,
    pub mean_trust: f64,
    pub mean_confidence: f64,
    pub emergence_ratio: f64,
    pub hubs: Vec<String>,
    pub isolated: Vec<String>,
    /// The most recent trace, when one has been run.
    pub last_trace: Option<PipelineTrace>,
    /// Whether work is currently flowing through the mesh.
    pub active: bool,

    // --- the integration surface -------------------------------------------
    /// Version of the contract this mesh speaks.
    pub contract_version: &'static str,
    /// Commitment over the whole mesh right now.
    pub mesh_digest: String,
    /// How the mesh normalizes reported milliseconds and cost units.
    pub calibration: RuntimeCalibration,
    /// Which model is deciding, and whether it is the one that was asked for.
    pub head: DecisionHead,
    /// Runs planned but not yet finalized.
    pub open_runs: usize,
    /// The benchmark, when one has been run. Computed once and carried, because
    /// re-running six policies over a full batch every frame would make the
    /// observatory a benchmark harness with a display attached.
    pub benchmark: Option<BenchmarkReport>,
}

/// Rescale a slice into `[0, 1]` against its own maximum.
///
/// A cohort where every agent scores the same is mapped to all-ones rather than
/// all-zeros: identical agents are equally strong, not uniformly worthless.
fn normalize(values: &[f64]) -> Vec<f64> {
    let max = values.iter().copied().fold(0.0_f64, f64::max);
    if max <= f64::EPSILON {
        return vec![0.0; values.len()];
    }
    values
        .iter()
        .map(|value| (value / max).clamp(0.0, 1.0))
        .collect()
}

impl Observation {
    /// Read the mesh. `features` is the stimulus used to compute live activation.
    pub fn capture(
        mesh: &NeuralMesh,
        features: &[f64],
        active: bool,
        last_trace: Option<PipelineTrace>,
    ) -> Self {
        Self::capture_with(
            mesh,
            features,
            active,
            last_trace,
            RuntimeCalibration::default(),
        )
    }

    /// Read the mesh against an explicit calibration.
    ///
    /// The calibration decides what a reported millisecond is worth on the `[0, 1]`
    /// scale the equations use, so the observatory has to be told the same one the
    /// adapter is using or the economics panel quotes numbers nothing else agrees with.
    pub fn capture_with(
        mesh: &NeuralMesh,
        features: &[f64],
        active: bool,
        last_trace: Option<PipelineTrace>,
        calibration: RuntimeCalibration,
    ) -> Self {
        let capabilities = mesh.capabilities();
        let capability_norm = normalize(&capabilities);

        let seeds = mesh.seed_activations(features);
        let activations = mesh.message_pass(&seeds);
        let activation_norm = normalize(&activations);

        let agents = mesh
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let (expected_latency_ms, expected_cost) =
                    router::expected_profile(node, &calibration);
                AgentView {
                    id: node.id.clone(),
                    label: node.label.clone(),
                    role: node.role.as_str().to_string(),
                    capability: capabilities.get(index).copied().unwrap_or(0.0),
                    capability_norm: capability_norm.get(index).copied().unwrap_or(0.0),
                    confidence: node.confidence.clamp(0.0, 1.0),
                    trust: mesh.trust.inbound_trust(&node.id).clamp(0.0, 1.0),
                    activation: activation_norm.get(index).copied().unwrap_or(0.0),
                    attempts: node.telemetry.attempts,
                    successes: node.telemetry.successes,
                    failures: node.telemetry.failures,
                    last_action: node.telemetry.last_action.clone(),
                    efficiency: node.resource_efficiency(),
                    expected_latency_ms,
                    expected_cost,
                    observations: node.resources.observations,
                }
            })
            .collect();

        let intelligence = mesh.intelligence();

        Self {
            agents,
            edges: mesh.edges(),
            emergence_ratio: intelligence.emergence_ratio(),
            intelligence,
            omega: mesh.global.omega,
            epoch: mesh.global.epoch,
            mean_trust: mesh.trust.mean_trust().clamp(0.0, 1.0),
            mean_confidence: mesh.mean_confidence().clamp(0.0, 1.0),
            hubs: mesh.trust.hubs(),
            isolated: mesh.trust.isolated(),
            last_trace,
            active,
            contract_version: CONTRACT_VERSION,
            mesh_digest: mesh.digest(),
            calibration,
            head: mesh.head(),
            open_runs: 0,
            benchmark: None,
        }
    }

    /// Attach a benchmark report for the benchmark tab.
    pub fn with_benchmark(mut self, benchmark: Option<BenchmarkReport>) -> Self {
        self.benchmark = benchmark;
        self
    }

    /// Record how many runs are planned but not yet finalized.
    pub fn with_open_runs(mut self, open_runs: usize) -> Self {
        self.open_runs = open_runs;
        self
    }

    /// Where the last run's outcomes came from, if there was one.
    pub fn provenance(&self) -> Option<OutcomeProvenance> {
        self.last_trace.as_ref().map(|trace| trace.provenance)
    }

    /// Mean expected cost per stage across the cohort.
    pub fn mean_expected_cost(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents
            .iter()
            .map(|agent| agent.expected_cost)
            .sum::<f64>()
            / self.agents.len() as f64
    }

    /// Mean expected latency per stage across the cohort, in milliseconds.
    pub fn mean_expected_latency_ms(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents
            .iter()
            .map(|agent| agent.expected_latency_ms)
            .sum::<f64>()
            / self.agents.len() as f64
    }

    /// How much of the cohort has been measured rather than merely declared.
    ///
    /// This is the honest headline for an integration: a mesh where nothing has been
    /// reported is running entirely on the numbers someone typed in at registration.
    pub fn measured_share(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents
            .iter()
            .filter(|agent| agent.is_measured())
            .count() as f64
            / self.agents.len() as f64
    }

    /// Mean activation across the cohort — the mesh's live intensity.
    pub fn intensity(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        let total: f64 = self.agents.iter().map(|agent| agent.activation).sum();
        (total / self.agents.len() as f64).clamp(0.0, 1.0)
    }

    /// `Ω` squashed into `[0, 1]` for display.
    ///
    /// `Ω` is unbounded above, so a linear axis would flatten out as the mesh
    /// improves. The saturating curve keeps early movement legible while never
    /// pinning at the top.
    pub fn omega_norm(&self) -> f64 {
        let omega = self.omega.max(0.0);
        (omega / (omega + 2.0)).clamp(0.0, 1.0)
    }

    /// Mesh intelligence `A` squashed into `[0, 1]` for display.
    ///
    /// Like `Ω`, `A` is unbounded above, so it needs a scale before it can go on
    /// a meter. The reference point is the cohort itself: `raw` equals the agent
    /// count when every agent is at full capability, so dividing by `total +
    /// count` puts a flawless disconnected cohort at exactly one half and leaves
    /// the upper half of the meter to the emergent term. A mesh that reads above
    /// `0.5` is worth more than the sum of its agents.
    pub fn intelligence_norm(&self) -> f64 {
        let total = self.intelligence.total.max(0.0);
        let reference = self.agents.len().max(1) as f64;
        (total / (total + reference)).clamp(0.0, 1.0)
    }

    /// Build the vortex for this observation.
    pub fn tornado(&self) -> Tornado {
        let nodes = self
            .agents
            .iter()
            .enumerate()
            .map(|(index, agent)| VortexNode {
                id: agent.id.clone(),
                label: agent.label.clone(),
                capability: agent.capability_norm,
                trust: agent.trust,
                activation: agent.activation,
                // Spread the cohort evenly around the funnel by construction.
                phase: index as f64 * std::f64::consts::TAU / self.agents.len().max(1) as f64,
            })
            .collect();

        Tornado {
            nodes,
            amplitude: self.omega_norm(),
            intensity: self.intensity(),
            active: self.active,
        }
    }

    /// The cohort ordered strongest first.
    pub fn ranked(&self) -> Vec<&AgentView> {
        let mut ranked: Vec<&AgentView> = self.agents.iter().collect();
        ranked.sort_by(|a, b| b.capability.total_cmp(&a.capability));
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::TaskSpec;
    use hatcher_neural::pipeline;

    fn features() -> Vec<f64> {
        vec![0.3, 0.7, 0.2, 0.9]
    }

    #[test]
    fn capture_reads_the_default_cohort() {
        let mesh = NeuralMesh::default();
        let observation = Observation::capture(&mesh, &features(), false, None);

        assert!(
            !observation.agents.is_empty(),
            "the default cohort must be populated"
        );
        assert_eq!(observation.agents.len(), mesh.nodes.len());
        assert!(observation.omega > 0.0);
        assert_eq!(observation.epoch, 0);
    }

    #[test]
    fn every_normalized_channel_is_inside_the_unit_interval() {
        let mesh = NeuralMesh::default();
        let observation = Observation::capture(&mesh, &features(), true, None);

        for agent in &observation.agents {
            for value in [
                agent.capability_norm,
                agent.activation,
                agent.trust,
                agent.confidence,
            ] {
                assert!(
                    (0.0..=1.0).contains(&value),
                    "{} out of range: {value}",
                    agent.id
                );
            }
        }
        assert!((0.0..=1.0).contains(&observation.intensity()));
        assert!((0.0..=1.0).contains(&observation.omega_norm()));
    }

    #[test]
    fn omega_norm_is_monotone_and_never_saturates() {
        let mut mesh = NeuralMesh::default();
        let low = Observation::capture(&mesh, &features(), false, None).omega_norm();

        mesh.global.omega = 25.0;
        let high = Observation::capture(&mesh, &features(), false, None).omega_norm();

        assert!(high > low);
        assert!(
            high < 1.0,
            "the curve must leave headroom above any finite omega"
        );
    }

    #[test]
    fn normalize_treats_a_uniform_cohort_as_uniformly_strong() {
        assert_eq!(normalize(&[2.0, 2.0, 2.0]), vec![1.0, 1.0, 1.0]);
        assert_eq!(normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
        assert_eq!(normalize(&[]), Vec::<f64>::new());
    }

    #[test]
    fn ranking_puts_the_strongest_agent_first() {
        let mesh = NeuralMesh::default();
        let observation = Observation::capture(&mesh, &features(), false, None);
        let ranked = observation.ranked();
        for pair in ranked.windows(2) {
            assert!(pair[0].capability >= pair[1].capability);
        }
    }

    #[test]
    fn running_a_task_advances_what_the_observatory_reports() {
        let mut mesh = NeuralMesh::default();
        let before = Observation::capture(&mesh, &features(), false, None);

        let task = TaskSpec::new("t-1", "ship the observatory", "rust")
            .with_features(features())
            .parsed_from_features();
        let trace = pipeline::run(&mut mesh, &task);

        let after = Observation::capture(&mesh, &features(), true, Some(trace));
        assert_eq!(after.epoch, before.epoch + 1);
        assert!(after.last_trace.is_some());
        assert_eq!(after.last_trace.as_ref().unwrap().stages.len(), 10);
    }

    #[test]
    fn the_tornado_inherits_the_cohort_and_its_intensity() {
        let mesh = NeuralMesh::default();
        let observation = Observation::capture(&mesh, &features(), true, None);
        let tornado = observation.tornado();

        assert_eq!(tornado.nodes.len(), observation.agents.len());
        assert!(tornado.active);
        assert!((tornado.intensity - observation.intensity()).abs() < 1e-12);
        assert!((tornado.amplitude - observation.omega_norm()).abs() < 1e-12);
    }
}
