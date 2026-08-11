//! Rehearsal arena for the agent mesh.
//!
//! The mesh only shows its character over many runs, so this crate exists to generate
//! histories: batches of tasks with controlled difficulty, run against a cohort, with
//! the resulting structure reported. Nothing here adds new dynamics — it drives
//! [`hatcher_neural::pipeline`] and reads the mesh's own state back out.
//!
//! [`bench`] answers the question the rest of the crate raises: run identical work under
//! different routing policies and see whether the mesh's routing actually beats a simple
//! fixed assignment on quality, cost, latency, and reliability — and replay real recorded
//! runs to check the same thing against what actually happened.

pub mod bench;

pub use bench::{
    contested_cohort, synthetic_recording, Benchmark, BenchmarkReport, PolicyDelta, PolicyScorecard,
    RecordedRun, Replay, ReplayReport, RoutingPolicy, StageAgreement,
};

use hatcher_core::{
    AgentNode, AgentRole, CapabilityVector, ExecutionMode, HatcherRequest, HatcherResponse, MeshCoefficients,
    MeshIntelligence, MeshSimulation, ModelSpec, PipelineTrace, ResourceProfile, TaskResult, TaskSpec,
    TaskSubmission,
};
use hatcher_neural::{default_cohort, pipeline, BackendKind, NeuralMesh};
use serde::{Deserialize, Serialize};

/// A live mesh plus the bookkeeping needed to submit work to it.
#[derive(Debug, Clone)]
pub struct AgentArena {
    pub mesh: NeuralMesh,
}

impl Default for AgentArena {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentArena {
    /// An arena over the default five-station cohort.
    pub fn new() -> Self {
        Self {
            mesh: NeuralMesh::default(),
        }
    }

    /// An arena over an explicit cohort.
    pub fn with_cohort(nodes: Vec<AgentNode>) -> Self {
        Self {
            mesh: NeuralMesh::with_cohort(nodes),
        }
    }

    /// An arena tuned with a specific coefficient set.
    pub fn with_coefficients(coefficients: MeshCoefficients) -> Self {
        Self {
            mesh: NeuralMesh::default().with_coefficients(coefficients),
        }
    }

    /// Load an ONNX decision head from `HATCHER_MESH_MODEL`, if that variable is set.
    ///
    /// A failed load degrades to the built-in head and prints why, rather than refusing
    /// to start: an operator who mistyped a path should get a working mesh and a clear
    /// warning, not a process that will not boot.
    pub fn with_policy_from_env(mut self) -> Self {
        let Ok(path) = std::env::var("HATCHER_MESH_MODEL") else {
            return self;
        };
        if path.trim().is_empty() {
            return self;
        }

        self.mesh = std::mem::take(&mut self.mesh).with_backend(
            BackendKind::Onnx {
                model_path: path.clone(),
            },
            ModelSpec::native_default(),
        );
        match self.mesh.degraded() {
            Some(reason) => eprintln!("warning: could not load `{path}`, using the built-in head: {reason}"),
            None => println!("loaded decision head from {path}"),
        }
        self
    }

    /// Run one bridge request through the pipeline, returning pretty JSON.
    pub fn run_round(&mut self, agent_id: &str, prompt: &str, features: Vec<f64>) -> String {
        let request = HatcherRequest::new(agent_id, AgentRole::Explorer, ExecutionMode::Sandbox)
            .with_prompt(prompt)
            .with_features(features);

        self.mesh
            .evaluate(&request)
            .to_json()
            .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
    }

    /// Run one bridge request and return the typed response.
    pub fn evaluate(&mut self, agent_id: &str, prompt: &str, features: Vec<f64>) -> HatcherResponse {
        let request = HatcherRequest::new(agent_id, AgentRole::Explorer, ExecutionMode::Controlled)
            .with_prompt(prompt)
            .with_features(features);
        self.mesh.evaluate(&request)
    }

    /// Submit a task exactly as the frontend would.
    pub fn submit(&mut self, submission: &TaskSubmission) -> TaskResult {
        let task = submission.to_task(self.mesh.sequence + 1);
        self.run_task(&task)
    }

    /// Run a task and package the trace with the resulting mesh headline.
    pub fn run_task(&mut self, task: &TaskSpec) -> TaskResult {
        let trace = pipeline::run(&mut self.mesh, task);
        TaskResult {
            trace,
            omega: self.mesh.global.omega,
            regime: self.mesh.global.regime(),
            intelligence: self.mesh.intelligence(),
        }
    }

    /// Rehearse without mutating the mesh, as serializable JSON.
    pub fn run_simulation(
        &self,
        agent_id: &str,
        prompt: &str,
        features: Vec<f64>,
        steps: usize,
    ) -> serde_json::Value {
        let request = HatcherRequest::new(agent_id, AgentRole::Explorer, ExecutionMode::Controlled)
            .with_prompt(prompt)
            .with_features(features);
        serde_json::to_value(self.rehearse(&request, steps)).unwrap_or(serde_json::Value::Null)
    }

    /// Rehearse without mutating the mesh.
    pub fn rehearse(&self, request: &HatcherRequest, steps: usize) -> MeshSimulation {
        self.mesh.rehearse(request, steps)
    }
}

/// A batch of comparable tasks at a controlled difficulty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub domain: String,
    pub tasks: usize,
    /// Drives both uncertainty `U` and implementation cost `I`, in `[0, 1]`.
    pub difficulty: f64,
    pub urgency: f64,
    pub execution_mode: ExecutionMode,
}

impl Scenario {
    pub fn new(name: impl Into<String>, domain: impl Into<String>, tasks: usize, difficulty: f64) -> Self {
        Self {
            name: name.into(),
            domain: domain.into(),
            tasks,
            difficulty: difficulty.clamp(0.0, 1.0),
            urgency: 0.5,
            execution_mode: ExecutionMode::Controlled,
        }
    }

    /// Routine work a competent mesh should handle.
    pub fn rehearsal() -> Self {
        Self::new("rehearsal", "rust", 24, 0.25)
    }

    /// Work at the edge of what the cohort can do.
    pub fn stress() -> Self {
        let mut scenario = Self::new("stress", "rust", 24, 0.85);
        scenario.urgency = 0.9;
        scenario
    }

    /// Work in a domain nobody has mastered, to exercise specialization growth.
    pub fn frontier() -> Self {
        Self::new("frontier", "distributed-systems", 24, 0.6)
    }

    /// Materialize the task batch. Ids and features are deterministic, so two runs of
    /// the same scenario are directly comparable.
    pub fn task_batch(&self) -> Vec<TaskSpec> {
        (0..self.tasks)
            .map(|index| {
                let phase = index as f64 / self.tasks.max(1) as f64;
                let mut task = TaskSpec::new(
                    format!("{}-{index:03}", self.name),
                    format!("{} task {index}", self.name),
                    self.domain.clone(),
                )
                .with_features(vec![phase, 1.0 - phase, self.difficulty, self.urgency])
                .with_execution_mode(self.execution_mode);

                task.uncertainty = self.difficulty;
                task.implementation_cost = self.difficulty;
                task.budget = 0.4 + 0.4 * self.difficulty;
                task.urgency = self.urgency;
                task
            })
            .collect()
    }
}

/// What a scenario did to the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioReport {
    pub name: String,
    pub tasks: usize,
    pub omega_start: f64,
    pub omega_end: f64,
    /// Fraction of runs that passed verification.
    pub verified_rate: f64,
    /// Fraction of staffed stages that succeeded.
    pub stage_success_rate: f64,
    pub mean_trust: f64,
    pub intelligence: MeshIntelligence,
    pub emergence_ratio: f64,
    pub hubs: Vec<String>,
    pub isolated: Vec<String>,
    pub escalations: usize,
    pub total_cost: f64,
}

impl ScenarioReport {
    pub fn omega_gain(&self) -> f64 {
        self.omega_end - self.omega_start
    }

    /// One-line summary for the terminal.
    pub fn headline(&self) -> String {
        format!(
            "{:<10} omega {:.3} -> {:.3} ({:+.3}) | verified {:.0}% | stages {:.0}% | trust {:.2} | E {:.0}% | escalations {}",
            self.name,
            self.omega_start,
            self.omega_end,
            self.omega_gain(),
            self.verified_rate * 100.0,
            self.stage_success_rate * 100.0,
            self.mean_trust,
            self.emergence_ratio * 100.0,
            self.escalations
        )
    }
}

/// Run a scenario against a mesh, mutating it, and report what changed.
pub fn run_scenario(mesh: &mut NeuralMesh, scenario: &Scenario) -> (ScenarioReport, Vec<PipelineTrace>) {
    let omega_start = mesh.global.omega;
    let traces = pipeline::run_batch(mesh, &scenario.task_batch());

    let runs = traces.len().max(1) as f64;
    let verified = traces.iter().filter(|trace| trace.verified).count() as f64;
    let escalations = traces
        .iter()
        .filter(|trace| trace.decision.action == "escalate")
        .count();

    let staffed: Vec<&hatcher_core::StageRecord> = traces
        .iter()
        .flat_map(|trace| trace.stages.iter())
        .filter(|record| record.stage.is_staffed())
        .collect();
    let stage_success = staffed.iter().filter(|record| record.success).count() as f64;

    let intelligence = mesh.intelligence();
    let report = ScenarioReport {
        name: scenario.name.clone(),
        tasks: traces.len(),
        omega_start,
        omega_end: mesh.global.omega,
        verified_rate: verified / runs,
        stage_success_rate: if staffed.is_empty() {
            0.0
        } else {
            stage_success / staffed.len() as f64
        },
        mean_trust: mesh.trust.mean_trust(),
        intelligence,
        emergence_ratio: intelligence.emergence_ratio(),
        hubs: mesh.trust.hubs(),
        isolated: mesh.trust.isolated(),
        escalations,
        total_cost: traces.iter().map(|trace| trace.total_cost()).sum(),
    };

    (report, traces)
}

/// A contender in a coefficient battle: a name and a tuning.
#[derive(Debug, Clone)]
pub struct Contender {
    pub name: String,
    pub coefficients: MeshCoefficients,
}

/// Standing of one contender after the battle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Standing {
    pub name: String,
    pub omega: f64,
    pub verified_rate: f64,
    pub mean_trust: f64,
    pub emergence_ratio: f64,
}

/// Result of a battle between coefficient tunings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleReport {
    pub winner: String,
    pub rounds: usize,
    pub summary: String,
    pub standings: Vec<Standing>,
}

/// Run identical work against differently-tuned meshes and see which ends up smartest.
///
/// This is the honest version of an agent battle: the cohorts are identical and the
/// tasks are identical, so the only variable is the coefficient set. What wins is a
/// tuning, not a lucky agent.
#[derive(Debug, Clone)]
pub struct AgentBattle {
    pub scenario: Scenario,
    pub contenders: Vec<Contender>,
}

impl Default for AgentBattle {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBattle {
    pub fn new() -> Self {
        Self {
            scenario: Scenario::rehearsal(),
            contenders: vec![
                Contender {
                    name: "conservative".into(),
                    coefficients: MeshCoefficients::conservative(),
                },
                Contender {
                    name: "balanced".into(),
                    coefficients: MeshCoefficients::default(),
                },
                Contender {
                    name: "exploratory".into(),
                    coefficients: MeshCoefficients::exploratory(),
                },
            ],
        }
    }

    pub fn with_scenario(mut self, scenario: Scenario) -> Self {
        self.scenario = scenario;
        self
    }

    pub fn run(&self) -> BattleReport {
        let mut standings = Vec::with_capacity(self.contenders.len());

        for contender in &self.contenders {
            let mut mesh = NeuralMesh::with_cohort(default_cohort()).with_coefficients(contender.coefficients);
            let (report, _) = run_scenario(&mut mesh, &self.scenario);
            standings.push(Standing {
                name: contender.name.clone(),
                omega: report.omega_end,
                verified_rate: report.verified_rate,
                mean_trust: report.mean_trust,
                emergence_ratio: report.emergence_ratio,
            });
        }

        standings.sort_by(|a, b| b.omega.partial_cmp(&a.omega).unwrap_or(std::cmp::Ordering::Equal));
        let winner = standings
            .first()
            .map(|standing| standing.name.clone())
            .unwrap_or_else(|| "none".to_string());

        BattleReport {
            summary: format!(
                "{} tuning reached the highest omega over {} tasks of `{}`",
                winner, self.scenario.tasks, self.scenario.name
            ),
            winner,
            rounds: self.contenders.len(),
            standings,
        }
    }
}

/// A cohort with one deliberately unreliable member, for demonstrating isolation.
///
/// The weakness is placed in `intelligence` and `context` on purpose. Specialization,
/// performance, and memory all *grow* through the learning equations, so an agent
/// handicapped only there would train its way out of the handicap within a couple of
/// dozen runs. Intelligence and context are the factors the pipeline does not lift, so
/// they are what a persistent weak link is actually made of.
pub fn cohort_with_a_weak_link() -> Vec<AgentNode> {
    default_cohort()
        .into_iter()
        .map(|node| {
            if node.role == AgentRole::Critic {
                node.with_capability(CapabilityVector::new(0.15, 0.5, 0.5, 0.20, 0.4))
                    .with_resources(ResourceProfile::new(0.8, 0.7))
                    .with_confidence(0.3)
            } else {
                node
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_arena_round_returns_a_decision() {
        let mut arena = AgentArena::new();
        let json = arena.run_round("guardian-01", "protect the policy envelope", vec![0.2, 0.5, 0.8, 0.3]);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(parsed["decision"]["action"].is_string());
        assert!(parsed["trace_id"].as_str().unwrap().starts_with("trace-"));
        assert_eq!(arena.mesh.global.epoch, 1);
    }

    #[test]
    fn submissions_run_like_frontend_traffic() {
        let mut arena = AgentArena::new();
        let submission = TaskSubmission {
            description: "refactor the trust graph".into(),
            domain: Some("rust".into()),
            features: vec![0.4, 0.6],
            urgency: Some(0.8),
            execution_mode: Some(ExecutionMode::Controlled),
        };

        let result = arena.submit(&submission);
        assert_eq!(result.trace.domain, "rust");
        assert_eq!(result.trace.stages.len(), 10);
        assert_eq!(result.omega, arena.mesh.global.omega);
    }

    #[test]
    fn simulation_is_serializable_and_leaves_the_arena_alone() {
        let arena = AgentArena::new();
        let value = arena.run_simulation("explorer", "rehearse", vec![0.3, 0.6], 3);
        assert_eq!(value["steps"].as_array().unwrap().len(), 3);
        assert_eq!(arena.mesh.global.epoch, 0);
    }

    #[test]
    fn scenario_batches_are_deterministic_and_correctly_sized() {
        let scenario = Scenario::rehearsal();
        let first = scenario.task_batch();
        let second = scenario.task_batch();

        assert_eq!(first.len(), 24);
        assert_eq!(first[0].id, "rehearsal-000");
        assert_eq!(first[0].uncertainty, scenario.difficulty);
        assert_eq!(
            first.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
            second.iter().map(|t| t.id.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_easy_scenario_leaves_the_mesh_smarter_than_a_brutal_one() {
        let mut easy_mesh = NeuralMesh::default();
        let (easy, _) = run_scenario(&mut easy_mesh, &Scenario::rehearsal());

        let mut hard_mesh = NeuralMesh::default();
        let (hard, _) = run_scenario(&mut hard_mesh, &Scenario::stress());

        assert!(
            easy.omega_end > hard.omega_end,
            "easy {:.3} should beat stress {:.3}",
            easy.omega_end,
            hard.omega_end
        );
        assert!(easy.verified_rate > hard.verified_rate);
        assert!(hard.escalations > 0, "stress work should reach a human");
        assert!(easy.total_cost > 0.0);
    }

    #[test]
    fn reports_summarize_without_panicking_on_an_empty_mesh() {
        let mut mesh = NeuralMesh::with_cohort(Vec::new());
        let (report, traces) = run_scenario(&mut mesh, &Scenario::new("void", "none", 3, 0.5));

        assert_eq!(traces.len(), 3);
        assert_eq!(report.stage_success_rate, 0.0);
        assert_eq!(report.verified_rate, 0.0);
        assert!(report.headline().contains("void"));
    }

    #[test]
    fn a_frontier_scenario_grows_new_domain_expertise() {
        let mut mesh = NeuralMesh::default();
        let before = mesh.node("coder-01").unwrap().mastery("distributed-systems");
        run_scenario(&mut mesh, &Scenario::frontier());
        let after = mesh.node("coder-01").unwrap().mastery("distributed-systems");
        assert!(after > before, "repeated work in a new domain must build mastery");
    }

    #[test]
    fn the_cohort_trusts_a_weak_critic_less_than_a_capable_one() {
        let scenario = Scenario::rehearsal();

        let mut weak = NeuralMesh::with_cohort(cohort_with_a_weak_link());
        run_scenario(&mut weak, &scenario);

        let mut healthy = NeuralMesh::default();
        run_scenario(&mut healthy, &scenario);

        assert!(
            weak.trust.trust_between("coder-01", "critic-01") < healthy.trust.trust_between("coder-01", "critic-01"),
            "identical work, identical tasks: the only difference is the critic, and the graph should show it \
             (weak={:.3} healthy={:.3})",
            weak.trust.trust_between("coder-01", "critic-01"),
            healthy.trust.trust_between("coder-01", "critic-01")
        );
        assert!(weak.global.omega < healthy.global.omega, "and a weak link costs the mesh omega");
    }

    #[test]
    fn a_battle_ranks_every_tuning_and_names_a_winner() {
        let report = AgentBattle::new().run();

        assert_eq!(report.rounds, 3);
        assert_eq!(report.standings.len(), 3);
        assert!(!report.winner.is_empty());
        assert_eq!(report.standings[0].name, report.winner);
        assert!(
            report.standings[0].omega >= report.standings[2].omega,
            "standings must be sorted by omega"
        );
        assert!(report.summary.contains(&report.winner));
    }

    #[test]
    fn battles_are_reproducible() {
        assert_eq!(AgentBattle::new().run().winner, AgentBattle::new().run().winner);
    }
}
