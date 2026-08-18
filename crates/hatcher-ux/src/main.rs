//! Terminal UX and HTTP surface for the Hatcher agent mesh.
//!
//! Two faces onto the same live mesh:
//!
//! * An interactive console for driving and inspecting it locally.
//! * A JSON API shaped for the `hatcher-host-frontend` dashboard. That app is a
//!   Next.js client that talks to its backend only through `lib/api.ts` and renders an
//!   agent across thirteen tabs, so the endpoints here line up with the tabs that are
//!   actually about intelligence:
//!
//! | Frontend tab | Endpoint |
//! |---|---|
//! | Overview | `GET /api/mesh/overview` |
//! | Config | `GET`/`PUT /api/mesh/config` |
//! | Analytics | `GET /api/mesh/analytics` |
//! | Logs | `GET /api/mesh/logs` |
//! | Workflows / Chat | `POST /api/tasks` |
//! | — (graph view) | `GET /api/mesh/graph`, `GET /api/mesh/trust` |
//!
//! The mesh runs as a sidecar: the frontend's own backend stays at `:3001`, and this
//! process listens on `HATCHER_MESH_PORT` (default `3030`).

use std::env;
use std::io::{self, Write};
use std::net::Ipv4Addr;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use hatcher_core::{
    AgentRegistration, AgentRole, ApiRequest, ApiResponse, Assignment, ContractError,
    ExecutionMode, HatcherRequest, MeshCoefficients, PipelineStage, PipelineTrace,
    StageOutcomeReport, TaskEnvelope, TaskResult, TaskSubmission, CONTRACT_VERSION,
};
use hatcher_neural::{pipeline, router, MeshAdapter};
use hatcher_playground::{AgentArena, AgentBattle, Benchmark, Replay, Scenario};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::runtime::Runtime;
use warp::filters::BoxedFilter;
use warp::http::StatusCode;
use warp::reject::Reject;
use warp::Filter;
use warp::Reply;

/// How many pipeline traces the Logs view keeps.
const TRACE_LOG_CAPACITY: usize = 200;

/// Default port for the mesh sidecar.
const DEFAULT_PORT: u16 = 3030;

/// Keep a compromised or misconfigured caller from turning one request into an
/// unbounded ranking job. Hatcher workspaces are far smaller than this today.
const MAX_SHADOW_AGENTS: usize = 128;

/// Version the narrow Hatcher routing envelopes independently from the full
/// register/plan/report/finalize integration contract. `shadow.v1` remains
/// available during the migration; Hatcher Live Mode uses `route.v2`.
const SHADOW_CONTRACT_VERSION: &str = "shadow.v1";
const ROUTING_CONTRACT_VERSION: &str = "route.v2";

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<MeshSession>>,
}

struct MeshSession {
    adapter: MeshAdapter,
    traces: Vec<PipelineTrace>,
    execution_mode: ExecutionMode,
}

impl MeshSession {
    fn new() -> Self {
        // The arena is only used to resolve `HATCHER_MESH_MODEL` — it owns the ONNX
        // fallback logic — and then hands its mesh to the adapter, which is what
        // everything else in this process drives.
        let arena = AgentArena::new().with_policy_from_env();
        Self {
            adapter: MeshAdapter::with_mesh(arena.mesh),
            traces: Vec::new(),
            execution_mode: ExecutionMode::Controlled,
        }
    }

    fn log(&mut self, trace: PipelineTrace) {
        if self.traces.len() == TRACE_LOG_CAPACITY {
            self.traces.remove(0);
        }
        self.traces.push(trace);
    }

    /// Submit a task the way the frontend does: simulated outcomes, one call.
    fn submit(&mut self, submission: &TaskSubmission) -> TaskResult {
        let task = submission.to_task(self.adapter.mesh.sequence + 1);
        let trace = pipeline::run(&mut self.adapter.mesh, &task);
        TaskResult {
            trace,
            omega: self.adapter.mesh.global.omega,
            regime: self.adapter.mesh.global.regime(),
            intelligence: self.adapter.mesh.intelligence(),
        }
    }
}

/// Turn a contract error into the status the contract says it maps to.
fn contract_reply(error: ContractError) -> Box<dyn warp::Reply> {
    let status = StatusCode::from_u16(error.status()).unwrap_or(StatusCode::BAD_REQUEST);
    Box::new(warp::reply::with_status(
        warp::reply::json(&json!({ "error": error.to_string(), "detail": error })),
        status,
    ))
}

fn contract_result<T: Serialize>(result: Result<T, ContractError>) -> Box<dyn warp::Reply> {
    match result {
        Ok(value) => Box::new(warp::reply::json(&value)),
        Err(error) => contract_reply(error),
    }
}

impl AppState {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MeshSession::new())),
        }
    }

    /// Run a closure against the session. Poisoned locks are recovered rather than
    /// propagated: one panicked request must not take the whole mesh offline.
    fn with<T>(&self, action: impl FnOnce(&mut MeshSession) -> T) -> T {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        action(&mut guard)
    }
}

/// Compact log row for the Logs tab — a whole trace is too heavy for a list view.
#[derive(Debug, Clone, Serialize)]
struct TraceSummary {
    task_id: String,
    domain: String,
    action: String,
    verified: bool,
    priority: f64,
    band: String,
    omega_before: f64,
    omega_after: f64,
    stages_failed: usize,
    total_cost: f64,
    digest: String,
}

/// Stateless, tenant-local routing request used by the Hatcher control plane in shadow
/// mode. The cohort exists only for this request, so agents from different owners can
/// never enter the same candidate pool.
#[derive(Debug, Clone, Deserialize)]
struct ShadowRouteRequest {
    agents: Vec<AgentRegistration>,
    task: TaskEnvelope,
}

#[derive(Debug, Clone, Serialize)]
struct ShadowCandidate {
    agent_id: String,
    score: f64,
    capability: f64,
    inbound_trust: f64,
    mastery: f64,
    expected_latency_ms: f64,
    expected_cost: f64,
    within_constraints: bool,
}

impl ShadowCandidate {
    fn from_assignment(
        assignment: Assignment,
        adapter: &MeshAdapter,
        task: &hatcher_core::TaskSpec,
    ) -> Self {
        let node = adapter
            .mesh
            .node(&assignment.agent_id)
            .expect("ranked assignments always refer to a registered node");
        let (expected_latency_ms, expected_cost) =
            router::expected_profile(node, &adapter.calibration);
        Self {
            agent_id: assignment.agent_id,
            score: assignment.score,
            capability: assignment.capability,
            inbound_trust: assignment.inbound_trust,
            mastery: assignment.mastery,
            expected_latency_ms,
            expected_cost,
            within_constraints: task.constraints.admits(expected_latency_ms, expected_cost),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ShadowRouteResponse {
    contract_version: &'static str,
    recommended_agent_id: Option<String>,
    confidence: f64,
    domain: String,
    band: String,
    mesh_digest: String,
    candidates: Vec<ShadowCandidate>,
}

fn build_route(
    request: ShadowRouteRequest,
    contract_version: &'static str,
) -> Result<ShadowRouteResponse, ContractError> {
    if request.agents.is_empty() {
        return Err(ContractError::invalid(
            "agents",
            "at least one tenant-owned agent is required",
        ));
    }
    if request.agents.len() > MAX_SHADOW_AGENTS {
        return Err(ContractError::invalid(
            "agents",
            format!("at most {MAX_SHADOW_AGENTS} tenant-owned agents are allowed"),
        ));
    }
    request.task.validate()?;

    let mut adapter = MeshAdapter::empty();
    for registration in request.agents {
        adapter.register_agent(registration)?;
    }

    let task = request.task.to_task(1);
    let band = adapter.mesh.priority_for(&task).band;
    let ranked = router::rank_with(
        &adapter.mesh,
        PipelineStage::Code,
        &task,
        band,
        &[],
        &adapter.calibration,
    );
    let confidence = match ranked.as_slice() {
        [] => 0.0,
        [_] => 1.0,
        [first, second, ..] => {
            let denominator = first.score + second.score;
            if denominator <= f64::EPSILON {
                0.5
            } else {
                (first.score / denominator).clamp(0.5, 1.0)
            }
        }
    };
    let recommended_agent_id = ranked.first().map(|candidate| candidate.agent_id.clone());
    let candidates = ranked
        .into_iter()
        .map(|assignment| ShadowCandidate::from_assignment(assignment, &adapter, &task))
        .collect();

    Ok(ShadowRouteResponse {
        contract_version,
        recommended_agent_id,
        confidence,
        domain: task.domain,
        band: band.as_str().to_string(),
        mesh_digest: adapter.mesh.digest(),
        candidates,
    })
}

fn build_shadow_route(request: ShadowRouteRequest) -> Result<ShadowRouteResponse, ContractError> {
    build_route(request, SHADOW_CONTRACT_VERSION)
}

fn build_live_route(request: ShadowRouteRequest) -> Result<ShadowRouteResponse, ContractError> {
    build_route(request, ROUTING_CONTRACT_VERSION)
}

fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    let length = left.len().max(right.len());
    for index in 0..length {
        difference |= usize::from(left.get(index).copied().unwrap_or_default())
            ^ usize::from(right.get(index).copied().unwrap_or_default());
    }
    difference == 0
}

fn token_policy_allows(expected: Option<&str>, required: bool, provided: Option<&str>) -> bool {
    match expected.filter(|token| !token.is_empty()) {
        Some(expected) => provided
            .map(|provided| constant_time_equal(expected.as_bytes(), provided.as_bytes()))
            .unwrap_or(false),
        None => !required,
    }
}

fn internal_token_allows(provided: Option<&str>) -> bool {
    let expected = env::var("HATCHER_MESH_INTERNAL_TOKEN").ok();
    token_policy_allows(
        expected.as_deref(),
        env_flag("HATCHER_MESH_REQUIRE_INTERNAL_TOKEN"),
        provided,
    )
}

fn mesh_bind_addr() -> Result<Ipv4Addr, String> {
    env::var("HATCHER_MESH_BIND_ADDR")
        .unwrap_or_else(|_| Ipv4Addr::LOCALHOST.to_string())
        .parse::<Ipv4Addr>()
        .map_err(|_| "HATCHER_MESH_BIND_ADDR must be a valid IPv4 address".to_string())
}

fn bind_policy_allows(address: Ipv4Addr, non_loopback_allowed: bool) -> bool {
    address.is_loopback() || non_loopback_allowed
}

fn validate_serve_configuration() -> Result<(), String> {
    if env_flag("HATCHER_MESH_REQUIRE_INTERNAL_TOKEN")
        && env::var("HATCHER_MESH_INTERNAL_TOKEN")
            .map(|token| token.trim().len() < 32)
            .unwrap_or(true)
    {
        return Err(
            "HATCHER_MESH_INTERNAL_TOKEN must contain at least 32 characters when token enforcement is enabled"
                .to_string(),
        );
    }
    let bind_address = mesh_bind_addr()?;
    if !bind_policy_allows(
        bind_address,
        env_flag("HATCHER_MESH_ALLOW_NON_LOOPBACK_BIND"),
    ) {
        return Err(
            "non-loopback binding requires HATCHER_MESH_ALLOW_NON_LOOPBACK_BIND=true".to_string(),
        );
    }
    Ok(())
}

#[derive(Debug)]
struct UnauthorizedShadowRequest;

impl Reject for UnauthorizedShadowRequest {}

async fn require_shadow_token(token: Option<String>) -> Result<(), warp::Rejection> {
    if internal_token_allows(token.as_deref()) {
        Ok(())
    } else {
        Err(warp::reject::custom(UnauthorizedShadowRequest))
    }
}

async fn recover_shadow_authentication(
    rejection: warp::Rejection,
) -> Result<Box<dyn Reply>, warp::Rejection> {
    if rejection.find::<UnauthorizedShadowRequest>().is_some() {
        return Ok(Box::new(warp::reply::with_status(
            warp::reply::json(&json!({ "error": "unauthorized" })),
            StatusCode::UNAUTHORIZED,
        )));
    }
    Err(rejection)
}

impl From<&PipelineTrace> for TraceSummary {
    fn from(trace: &PipelineTrace) -> Self {
        Self {
            task_id: trace.task_id.clone(),
            domain: trace.domain.clone(),
            action: trace.decision.action.clone(),
            verified: trace.verified,
            priority: trace.priority.value,
            band: trace.priority.band.as_str().to_string(),
            omega_before: trace.omega_before,
            omega_after: trace.omega_after,
            stages_failed: trace.failures().len(),
            total_cost: trace.total_cost(),
            digest: trace.digest.chars().take(16).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP API
// ---------------------------------------------------------------------------

fn with_state(
    state: AppState,
) -> impl Filter<Extract = (AppState,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}

fn routes(state: AppState, shadow_only: bool) -> BoxedFilter<(warp::reply::Response,)> {
    let health = warp::path!("health")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            let (omega, agents, epoch) = state.with(|session| {
                (
                    session.adapter.mesh.global.omega,
                    session.adapter.mesh.nodes.len(),
                    session.adapter.mesh.global.epoch,
                )
            });
            warp::reply::json(&json!({
                "status": "ok",
                "service": "hatcher-mesh",
                "omega": omega,
                "agents": agents,
                "epoch": epoch,
            }))
        });

    let overview = warp::path!("api" / "mesh" / "overview")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.adapter.mesh.overview())));

    let analytics = warp::path!("api" / "mesh" / "analytics")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.adapter.mesh.analytics())));

    let graph = warp::path!("api" / "mesh" / "graph")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.adapter.mesh.graph_view())));

    let trust = warp::path!("api" / "mesh" / "trust")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.adapter.mesh.trust_view())));

    let agents = warp::path!("api" / "mesh" / "agents")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            warp::reply::json(&state.with(|s| s.adapter.mesh.agent_summaries()))
        });

    let get_config = warp::path!("api" / "mesh" / "config")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            warp::reply::json(&state.with(|s| {
                let mode = s.execution_mode;
                s.adapter.mesh.config_view(mode)
            }))
        });

    // Coefficients are validated before they are installed: a bad tuning would make
    // every subsequent equation diverge, and a 400 is far easier to debug than a mesh
    // that quietly stopped converging.
    let put_config = warp::path!("api" / "mesh" / "config")
        .and(warp::put().or(warp::post()).unify())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(
            |coefficients: MeshCoefficients, state: AppState| -> Box<dyn warp::Reply> {
                match coefficients.validate() {
                    Ok(()) => {
                        let view = state.with(|session| {
                            session.adapter.mesh.coefficients = coefficients;
                            let mode = session.execution_mode;
                            session.adapter.mesh.config_view(mode)
                        });
                        Box::new(warp::reply::json(&view))
                    }
                    Err(error) => Box::new(warp::reply::with_status(
                        warp::reply::json(&json!({ "error": error.to_string() })),
                        StatusCode::BAD_REQUEST,
                    )),
                }
            },
        );

    let logs = warp::path!("api" / "mesh" / "logs")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            let rows = state.with(|session| {
                session
                    .traces
                    .iter()
                    .rev()
                    .map(TraceSummary::from)
                    .collect::<Vec<_>>()
            });
            warp::reply::json(&rows)
        });

    let memory = warp::path!("api" / "mesh" / "memory")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.adapter.mesh.memory.clone())));

    let submit_task = warp::path!("api" / "tasks")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(|submission: TaskSubmission, state: AppState| {
            let result = state.with(|session| {
                let result = session.submit(&submission);
                session.log(result.trace.clone());
                result
            });
            warp::reply::json(&result)
        });

    let list_tasks = warp::path!("api" / "tasks")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            let rows = state.with(|session| {
                session
                    .traces
                    .iter()
                    .rev()
                    .map(TraceSummary::from)
                    .collect::<Vec<_>>()
            });
            warp::reply::json(&rows)
        });

    let trace_by_id = warp::path!("api" / "tasks" / String)
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|task_id: String, state: AppState| -> Box<dyn warp::Reply> {
            let found = state.with(|session| {
                session
                    .traces
                    .iter()
                    .rev()
                    .find(|trace| trace.task_id == task_id)
                    .cloned()
            });
            match found {
                Some(trace) => Box::new(warp::reply::json(&trace)),
                None => Box::new(warp::reply::with_status(
                    warp::reply::json(
                        &json!({ "error": format!("no trace for task `{task_id}`") }),
                    ),
                    StatusCode::NOT_FOUND,
                )),
            }
        });

    // Kept for the original bridge contract.
    let infer = warp::path!("api" / "infer")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(|request: ApiRequest, state: AppState| {
            let response = state.with(|session| {
                let bridge = HatcherRequest::new(
                    &request.agent_id,
                    AgentRole::Explorer,
                    session.execution_mode,
                )
                .with_prompt(&request.prompt)
                .with_features(request.features.clone());
                let hatcher = session.adapter.mesh.evaluate(&bridge);
                ApiResponse {
                    accepted: hatcher.accepted,
                    action: hatcher.decision.action.clone(),
                    confidence: hatcher.decision.confidence,
                    trace_id: hatcher.trace_id.clone(),
                }
            });
            warp::reply::json(&response)
        });

    // Rehearsal: read-only, so it can be hammered without moving the mesh.
    let simulate = warp::path!("api" / "simulate")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(|request: ApiRequest, state: AppState| {
            let value = state.with(|session| {
                let bridge = HatcherRequest::new(
                    &request.agent_id,
                    AgentRole::Explorer,
                    session.execution_mode,
                )
                .with_prompt(&request.prompt)
                .with_features(request.features.clone());
                serde_json::to_value(session.adapter.mesh.rehearse(&bridge, 5))
                    .unwrap_or(serde_json::Value::Null)
            });
            warp::reply::json(&value)
        });

    let battle = warp::path!("api" / "battle")
        .and(warp::get())
        .map(|| warp::reply::json(&AgentBattle::new().run()));

    // -----------------------------------------------------------------------
    // The integration contract
    // -----------------------------------------------------------------------

    // What a client needs to bind: the version, the verbs, and the scale the mesh
    // normalizes reported milliseconds and cost units against.
    let contract = warp::path!("api" / "contract")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            let (calibration, open, cohort, head) = state.with(|session| {
                (
                    session.adapter.calibration,
                    session.adapter.open_runs().len(),
                    session.adapter.mesh.nodes.len(),
                    session.adapter.head(),
                )
            });
            warp::reply::json(&json!({
                "contract_version": CONTRACT_VERSION,
                "calibration": calibration,
                "open_runs": open,
                "cohort_size": cohort,
                // A caller acting on `stabilize` is entitled to know which model proposed
                // it, and whether the mesh is running the one it was asked to run.
                "decision_head": head,
                "verbs": {
                    "register": "POST /api/mesh/agents",
                    "plan":     "POST /api/runs",
                    "report":   "POST /api/runs/{run_id}/outcomes",
                    "finalize": "POST /api/runs/{run_id}/finalize",
                },
            }))
        });

    // Routing is deliberately stateless and tenant-local. Hatcher supplies only the
    // current owner's eligible cohort and remains the sole execution authority. The
    // sidecar ranks; Hatcher decides whether a recommendation is shadowed or applied.
    let shadow_route = warp::path!("api" / "shadow" / "route")
        .and(warp::post())
        .and(warp::header::optional::<String>("x-hatcher-mesh-token"))
        .and_then(require_shadow_token)
        .and(warp::body::content_length_limit(256 * 1024))
        .and(warp::body::json())
        .map(|_: (), request: ShadowRouteRequest| contract_result(build_shadow_route(request)))
        .recover(recover_shadow_authentication);

    let live_route = warp::path!("api" / "route")
        .and(warp::post())
        .and(warp::header::optional::<String>("x-hatcher-mesh-token"))
        .and_then(require_shadow_token)
        .and(warp::body::content_length_limit(256 * 1024))
        .and(warp::body::json())
        .map(|_: (), request: ShadowRouteRequest| contract_result(build_live_route(request)))
        .recover(recover_shadow_authentication);

    let production_surface = health
        .clone()
        .or(contract.clone())
        .or(shadow_route)
        .or(live_route)
        .map(Reply::into_response)
        .boxed();

    let register_agent = warp::path!("api" / "mesh" / "agents")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(|registration: AgentRegistration, state: AppState| {
            contract_result(state.with(|session| session.adapter.register_agent(registration)))
        });

    let plan_run = warp::path!("api" / "runs")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(|envelope: TaskEnvelope, state: AppState| {
            contract_result(state.with(|session| session.adapter.plan(&envelope)))
        });

    let list_runs = warp::path!("api" / "runs")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            warp::reply::json(&state.with(|session| session.adapter.open_runs()))
        });

    let run_status = warp::path!("api" / "runs" / String)
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|run_id: String, state: AppState| {
            contract_result(state.with(|session| session.adapter.status(&run_id)))
        });

    let cancel_run = warp::path!("api" / "runs" / String)
        .and(warp::delete())
        .and(with_state(state.clone()))
        .map(|run_id: String, state: AppState| {
            contract_result(state.with(|session| {
                session
                    .adapter
                    .cancel(&run_id)
                    .map(|()| json!({ "cancelled": run_id }))
            }))
        });

    // A list rather than a single report, because a runtime that finishes a whole task
    // before calling home should not have to make five round trips to say so.
    let report_outcomes = warp::path!("api" / "runs" / String / "outcomes")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(
            |run_id: String, reports: Vec<StageOutcomeReport>, state: AppState| {
                contract_result(state.with(|session| session.adapter.report_many(&run_id, reports)))
            },
        );

    let finalize_run = warp::path!("api" / "runs" / String / "finalize")
        .and(warp::post())
        .and(with_state(state.clone()))
        .map(|run_id: String, state: AppState| {
            contract_result(state.with(|session| {
                let receipt = session.adapter.finalize(&run_id)?;
                if let Some(trace) = session.adapter.last_trace().cloned() {
                    session.log(trace);
                }
                Ok(receipt)
            }))
        });

    let benchmark = warp::path!("api" / "benchmark")
        .and(warp::get())
        .map(|| warp::reply::json(&Benchmark::new().run()));

    let replay = warp::path!("api" / "replay")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(|replay: Replay, state: AppState| {
            // Replayed against a copy of the cohort, never the live mesh: a replay is a
            // measurement, and one that moved the mesh it was measuring would be worth
            // nothing the second time it was run.
            let (cohort, calibration) = state.with(|session| {
                (
                    session.adapter.mesh.nodes.clone(),
                    session.adapter.calibration,
                )
            });
            warp::reply::json(&replay.evaluate(cohort, calibration))
        });

    let reset = warp::path!("api" / "mesh" / "reset")
        .and(warp::post())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            let overview = state.with(|session| {
                *session = MeshSession::new();
                session.adapter.mesh.overview()
            });
            warp::reply::json(&overview)
        });

    let allowed_origin = env::var("HATCHER_MESH_ALLOWED_ORIGIN").ok();
    let mut cors = warp::cors()
        .allow_headers(vec!["content-type", "authorization"])
        .allow_methods(vec!["GET", "POST", "PUT", "OPTIONS"]);
    cors = match allowed_origin.as_deref() {
        Some(origin) => cors.allow_origin(origin),
        // Convenient for local development against the Next.js dev server. Set
        // HATCHER_MESH_ALLOWED_ORIGIN before exposing this beyond localhost.
        None => cors.allow_any_origin(),
    };

    // Combined in three boxed groups rather than one long `.or()` chain. Each `.or()`
    // nests the filter's type one level deeper, and a single chain this long makes rustc
    // recurse far enough to blow its stack on Windows. `boxed()` erases the type at each
    // group boundary and keeps the nesting shallow.
    let read_models = health
        .or(overview)
        .or(analytics)
        .or(graph)
        .or(trust)
        .or(agents)
        .or(get_config)
        .or(put_config)
        .or(logs)
        .or(memory)
        .boxed();

    let tasks = submit_task
        .or(list_tasks)
        .or(trace_by_id)
        .or(infer)
        .or(simulate)
        .or(battle)
        .or(benchmark)
        .or(replay)
        .or(reset)
        .boxed();

    let contract_surface = contract
        .or(shadow_route)
        .or(live_route)
        .or(register_agent)
        // The two-segment run routes come before the one-segment ones: warp matches in
        // order, and `/api/runs/{id}` would otherwise swallow `/api/runs/{id}/finalize`.
        .or(report_outcomes)
        .or(finalize_run)
        .or(plan_run)
        .or(list_runs)
        .or(run_status)
        .or(cancel_run)
        .boxed();

    if shadow_only {
        production_surface
    } else {
        read_models
            .or(tasks)
            .or(contract_surface)
            .with(cors)
            .map(Reply::into_response)
            .boxed()
    }
}

fn mesh_port() -> u16 {
    env::var("HATCHER_MESH_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

async fn serve_http(state: AppState, port: u16) {
    if let Err(error) = validate_serve_configuration() {
        eprintln!("refusing to start mesh API: {error}");
        std::process::exit(78);
    }
    let shadow_only = env_flag("HATCHER_MESH_SHADOW_ONLY");
    let bind_address = mesh_bind_addr().expect("serve configuration was already validated");
    println!("mesh API listening on http://{bind_address}:{port}");
    println!("  GET  /health");
    println!("  GET  /api/mesh/overview | analytics | graph | trust | agents | logs | memory");
    println!("  GET  /api/mesh/config     PUT /api/mesh/config");
    println!("  POST /api/tasks           GET /api/tasks   GET /api/tasks/{{id}}");
    println!("  POST /api/infer | /api/simulate | /api/mesh/reset");
    println!("  GET  /api/battle | /api/benchmark    POST /api/replay");
    println!("  -- integration contract v{CONTRACT_VERSION} --");
    println!("  GET  /api/contract");
    println!("  POST /api/shadow/route                        rank one tenant-local cohort");
    println!("  POST /api/route                               rank for Hatcher Live Mode");
    println!("  POST /api/mesh/agents                          register an agent");
    println!("  POST /api/runs                                 plan a run");
    println!("  POST /api/runs/{{id}}/outcomes                   report what happened");
    println!("  POST /api/runs/{{id}}/finalize                   receive the decision");
    println!("  GET  /api/runs | /api/runs/{{id}}   DELETE /api/runs/{{id}}");
    if shadow_only {
        println!("  production surface: health, contract, and shadow routing only");
    }
    warp::serve(routes(state, shadow_only))
        .run((bind_address.octets(), port))
        .await;
}

// ---------------------------------------------------------------------------
// Terminal telemetry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TelemetrySnapshot {
    tick: usize,
    omega: f64,
    regime: String,
    agents: usize,
    intelligence: f64,
    emergence: f64,
    mean_trust: f64,
    hubs: usize,
    isolated: usize,
    action: String,
    verified: bool,
}

impl TelemetrySnapshot {
    fn render(&self) -> String {
        format!(
            "\x1b[2J\x1b[H\
╔════════════════ HatcherLabs Mesh Telemetry ═══════════════╗
║ Tick        : {:>3}                                        ║
║ Omega       : {:>6.3}  ({:<12})                      ║
║ Agents      : {:>3}                                        ║
║ Mesh A      : {:>6.3}  emergent {:>5.1}%                   ║
║ Mean trust  : {:>6.3}                                     ║
║ Hubs        : {:>3}     isolated {:>3}                       ║
║ Last action : {:<10} verified {:<5}                   ║
╚═══════════════════════════════════════════════════════════╝
",
            self.tick,
            self.omega,
            self.regime,
            self.agents,
            self.intelligence,
            self.emergence * 100.0,
            self.mean_trust,
            self.hubs,
            self.isolated,
            self.action,
            self.verified
        )
    }
}

fn build_telemetry_snapshot(state: &AppState, tick: usize) -> TelemetrySnapshot {
    state.with(|session| {
        // Each tick is a real task, so the numbers on screen are the mesh actually
        // working rather than a decorative animation.
        let submission = TaskSubmission {
            description: format!("telemetry probe {tick}"),
            domain: Some("rust".into()),
            features: vec![0.3, 0.6, 0.4, 0.7],
            urgency: Some(0.4),
            execution_mode: Some(ExecutionMode::Sandbox),
        };
        let result = session.submit(&submission);
        session.log(result.trace.clone());

        let overview = session.adapter.mesh.overview();
        TelemetrySnapshot {
            tick,
            omega: overview.omega,
            regime: overview.regime.as_str().to_string(),
            agents: overview.agent_count,
            intelligence: overview.intelligence.total,
            emergence: overview.emergence_ratio,
            mean_trust: session.adapter.mesh.trust.mean_trust(),
            hubs: overview.hubs.len(),
            isolated: overview.isolated.len(),
            action: result.trace.decision.action.clone(),
            verified: result.trace.verified,
        }
    })
}

fn run_live_telemetry(state: &AppState, ticks: usize) {
    println!("Streaming live mesh telemetry. Press Ctrl+C to stop.");
    for tick in 1..=ticks {
        let snapshot = build_telemetry_snapshot(state, tick);
        print!("{}", snapshot.render());
        let _ = io::stdout().flush();
        thread::sleep(Duration::from_millis(250));
    }
    println!("\nTelemetry stream complete.");
}

// ---------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------

fn print_menu() {
    println!("  run / r        - submit one task through the full pipeline");
    println!("  scenario / s   - run a 24-task scenario (rehearsal | stress | frontier)");
    println!("  battle / b     - compare coefficient tunings over identical work");
    println!("  contract / c   - drive one task through the integration contract");
    println!("  bench          - mesh routing vs the baselines (rehearsal | stress | frontier)");
    println!("  replay         - replay a recording and score routing against it");
    println!("  overview / o   - agent roster, hubs, isolated agents");
    println!("  trust / t      - trust matrix and strongest collaborations");
    println!("  omega / w      - omega ledger and its terms");
    println!("  memory / m     - memory graph");
    println!("  logs / l       - recent pipeline traces");
    println!("  telemetry / y  - stream live telemetry");
    println!("  api / a        - start the HTTP API for the Hatcher frontend");
    println!("  reset          - start from a fresh mesh");
    println!("  help / h       - show this menu");
    println!("  quit / q       - exit");
}

fn command_run(state: &AppState) {
    let trace = state.with(|session| {
        let submission = TaskSubmission {
            description: "protect the policy envelope while shipping the router".into(),
            domain: Some("rust".into()),
            features: vec![0.2, 0.5, 0.8, 0.3],
            urgency: Some(0.6),
            execution_mode: Some(session.execution_mode),
        };
        let result = session.submit(&submission);
        session.log(result.trace.clone());
        result.trace
    });

    println!(
        "task {} | {} | P={:.3} ({})",
        trace.task_id,
        trace.domain,
        trace.priority.value,
        trace.priority.band.as_str()
    );
    for record in &trace.stages {
        println!(
            "  {:<14} {:<14} {:<4} conf={:.2}  {}",
            record.stage.as_str(),
            record.agent.clone().unwrap_or_else(|| "-".into()),
            if record.success { "ok" } else { "FAIL" },
            record.confidence,
            record.note
        );
    }
    println!(
        "decision: {} (confidence {:.2}) | verified={} | omega {:.4} -> {:.4}",
        trace.decision.action,
        trace.decision.confidence,
        trace.verified,
        trace.omega_before,
        trace.omega_after
    );
    println!("digest: {}", trace.digest);
}

fn command_scenario(state: &AppState, name: &str) {
    let scenario = match name {
        "stress" => Scenario::stress(),
        "frontier" => Scenario::frontier(),
        _ => Scenario::rehearsal(),
    };

    let report = state.with(|session| {
        let (report, traces) =
            hatcher_playground::run_scenario(&mut session.adapter.mesh, &scenario);
        for trace in traces {
            session.log(trace);
        }
        report
    });

    println!("{}", report.headline());
    println!("hubs     : {:?}", report.hubs);
    println!("isolated : {:?}", report.isolated);
}

fn command_overview(state: &AppState) {
    let overview = state.with(|session| session.adapter.mesh.overview());
    println!(
        "omega {:.4} ({}) | epoch {} | A={:.4} (emergent {:.1}%) | {} agents, {} live edges",
        overview.omega,
        overview.regime.as_str(),
        overview.epoch,
        overview.intelligence.total,
        overview.emergence_ratio * 100.0,
        overview.agent_count,
        overview.active_edge_count
    );
    println!(
        "{:<14} {:<12} {:>6} {:>6} {:>6} {:>6} {:>5} {:>4}  bottleneck",
        "agent", "role", "A_i", "conf", "in_T", "R_i", "deg", "att"
    );
    for agent in &overview.agents {
        println!(
            "{:<14} {:<12} {:>6.3} {:>6.2} {:>6.2} {:>6.2} {:>5} {:>4}  {}{}{}",
            agent.id,
            agent.role,
            agent.capability,
            agent.confidence,
            agent.inbound_trust,
            agent.efficiency,
            agent.degree,
            agent.attempts,
            agent.bottleneck,
            if agent.hub { "  [hub]" } else { "" },
            if agent.isolated { "  [isolated]" } else { "" }
        );
    }
}

fn command_trust(state: &AppState) {
    let (view, collaborations) = state.with(|session| {
        (
            session.adapter.mesh.trust_view(),
            session.adapter.mesh.top_collaborations(5),
        )
    });

    print!("{:<14}", "T_ij");
    for id in &view.ids {
        print!("{:>10}", short(id));
    }
    println!();
    for (row, id) in view.ids.iter().enumerate() {
        print!("{:<14}", short(id));
        for column in 0..view.ids.len() {
            print!("{:>10.2}", view.trust[row][column]);
        }
        println!();
    }

    println!("\nstrongest collaborations (contribution to emergent A):");
    for (from, to, contribution) in collaborations {
        println!("  {from} -> {to}: {contribution:.5}");
    }
}

fn command_omega(state: &AppState) {
    let analytics = state.with(|session| session.adapter.mesh.analytics());
    println!(
        "omega {:.4} | slope {:+.5}/epoch | A={:.4} | mean trust {:.3} | mean conf {:.3} | tasks {} ({:.0}% ok)",
        analytics.omega,
        analytics.omega_slope,
        analytics.intelligence.total,
        analytics.mean_trust,
        analytics.mean_confidence,
        analytics.total_tasks,
        analytics.success_rate * 100.0
    );
    println!(
        "{:>6} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "epoch", "omega", "L", "E", "C", "F", "D"
    );
    for sample in analytics
        .series
        .iter()
        .rev()
        .take(15)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        println!(
            "{:>6} {:>8.4} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
            sample.epoch,
            sample.omega,
            sample.delta.learning,
            sample.delta.emergence,
            sample.delta.collaboration,
            sample.delta.failure,
            sample.delta.drift
        );
    }
}

fn command_logs(state: &AppState) {
    let rows = state.with(|session| {
        session
            .traces
            .iter()
            .rev()
            .take(15)
            .map(TraceSummary::from)
            .collect::<Vec<_>>()
    });

    if rows.is_empty() {
        println!("no traces yet; try `run` or `scenario`");
        return;
    }
    println!(
        "{:<18} {:<10} {:<10} {:<9} {:>7} {:>8}",
        "task", "action", "band", "verified", "P", "omega"
    );
    for row in rows {
        println!(
            "{:<18} {:<10} {:<10} {:<9} {:>7.3} {:>8.4}",
            row.task_id, row.action, row.band, row.verified, row.priority, row.omega_after
        );
    }
}

fn command_memory(state: &AppState) {
    let memory = state.with(|session| session.adapter.mesh.memory.clone());
    println!("nodes  : {}", memory.nodes.len());
    println!("edges  : {}", memory.edges.len());
    println!("records: {}", memory.records.len());
    for (domain, mass) in memory.domain_mass() {
        println!("  {domain:<24} salience {mass:.3}");
    }
}

fn short(id: &str) -> String {
    id.chars().take(9).collect()
}

/// Drive one task through the full integration contract, printing each verb.
///
/// The console command exists so the contract can be *seen* working before anyone wires
/// a real runtime to it — plan, report, finalize, with the receipt at the end.
fn command_contract(state: &AppState) {
    let envelope = TaskEnvelope::new("harden the trust settlement path")
        .with_domain("rust")
        .with_features(vec![0.35, 0.72, 0.28, 0.64])
        .with_urgency(0.6);

    let plan = match state.with(|session| session.adapter.plan(&envelope)) {
        Ok(plan) => plan,
        Err(error) => return println!("plan rejected: {error}"),
    };

    let head = state.with(|session| session.adapter.head());
    println!("contract v{}   run {}", plan.contract_version, plan.run_id);
    println!(
        "head     {} ({} v{}, {}→{}){}",
        head.name,
        head.model,
        head.version,
        head.input_dim,
        head.output_dim,
        match head.degraded.as_deref() {
            Some(reason) => format!("  ⚠ degraded: {reason}"),
            None => String::new(),
        }
    );
    println!(
        "plan     P={:.3} ({})  expects {:.0}ms / {:.3} cost",
        plan.priority.value,
        plan.band.as_str(),
        plan.expected_latency_ms,
        plan.expected_cost
    );
    for planned in &plan.stages {
        println!(
            "  {:<9} → {:<18} {}",
            planned.stage.as_str(),
            planned.agent_id,
            planned.rationale
        );
    }
    if !plan.constraint_violations.is_empty() {
        println!(
            "  ! could not satisfy constraints at: {}",
            plan.constraint_violations.join(", ")
        );
    }

    // Stand in for a real runtime: report every stage as a solid success.
    let reports: Vec<StageOutcomeReport> = plan
        .stages
        .iter()
        .map(|planned| {
            StageOutcomeReport::success(planned.stage, &planned.agent_id, 0.88)
                .with_latency_ms(planned.expected_latency_ms)
                .with_cost(planned.expected_cost)
        })
        .collect();

    match state.with(|session| session.adapter.report_many(&plan.run_id, reports)) {
        Ok(status) => println!(
            "report   {} of 5 stages in, complete={}",
            status.reported.len(),
            status.complete
        ),
        Err(error) => return println!("report rejected: {error}"),
    }

    match state.with(|session| {
        let receipt = session.adapter.finalize(&plan.run_id)?;
        if let Some(trace) = session.adapter.last_trace().cloned() {
            session.log(trace);
        }
        Ok::<_, ContractError>(receipt)
    }) {
        Ok(receipt) => {
            println!(
                "receipt  {} by {} (confidence {:.2}) | provenance {} | verified {}",
                receipt.decision.action,
                receipt.selected_agent.as_deref().unwrap_or("—"),
                receipt.decision.confidence,
                receipt.provenance.as_str(),
                receipt.verified
            );
            println!("         {}", receipt.decision.rationale);
            println!(
                "         Ω {:.4} → {:.4} | {:.0}ms | {:.3} cost | quality {:.2}",
                receipt.omega_before,
                receipt.omega_after,
                receipt.total_latency_ms,
                receipt.total_cost,
                receipt.mean_quality
            );
            println!("         trace {}", short_digest(&receipt.trace_digest));
        }
        Err(error) => println!("finalize rejected: {error}"),
    }
}

fn short_digest(digest: &str) -> String {
    digest.chars().take(16).collect()
}

/// Mesh routing against the baselines, over identical work.
fn command_benchmark(state: &AppState, argument: &str) {
    let scenario = match argument {
        "stress" => Scenario::stress(),
        "frontier" => Scenario::frontier(),
        _ => Scenario::rehearsal(),
    };
    let _ = state;
    println!("{}", Benchmark::new().with_scenario(scenario).run().table());
}

/// Replay a synthetic recording and score the mesh's routing against it.
fn command_replay(state: &AppState) {
    let replay = hatcher_playground::synthetic_recording(40, "expert");
    let (cohort, calibration) = state.with(|session| {
        (
            session.adapter.mesh.nodes.clone(),
            session.adapter.calibration,
        )
    });
    let report = replay.evaluate(cohort, calibration);

    println!("{}", report.headline());
    for (stage, agreement) in &report.per_stage_agreement {
        println!(
            "  {:<10} agreed {:>3}/{:<3} ({:.0}%)",
            stage,
            agreement.agreed,
            agreement.compared,
            agreement.rate() * 100.0
        );
    }
    println!(
        "  router added value: {}",
        if report.router_added_value() {
            "yes"
        } else {
            "not demonstrated"
        }
    );
}

fn interactive_cli(state: AppState) {
    println!("HatcherLabs agent mesh neural system");
    println!("====================================");
    print_menu();
    println!("Tip: pressing Enter runs one task.");

    loop {
        print!("\n> ");
        let _ = io::stdout().flush();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let line = input.trim().to_ascii_lowercase();
        let mut parts = line.split_whitespace();
        let command = parts.next().unwrap_or("run");
        let argument = parts.next().unwrap_or("");

        match command {
            "run" | "r" | "demo" | "d" | "" => command_run(&state),
            "scenario" | "s" => command_scenario(&state, argument),
            "battle" | "b" => {
                let report = AgentBattle::new().run();
                println!("{}", report.summary);
                for standing in &report.standings {
                    println!(
                        "  {:<14} omega {:.4} | verified {:.0}% | trust {:.2} | emergent {:.1}%",
                        standing.name,
                        standing.omega,
                        standing.verified_rate * 100.0,
                        standing.mean_trust,
                        standing.emergence_ratio * 100.0
                    );
                }
            }
            "contract" | "c" => command_contract(&state),
            "bench" => command_benchmark(&state, argument),
            "replay" => command_replay(&state),
            "overview" | "o" => command_overview(&state),
            "trust" | "t" => command_trust(&state),
            "omega" | "w" => command_omega(&state),
            "memory" | "m" => command_memory(&state),
            "logs" | "l" => command_logs(&state),
            "telemetry" | "y" => run_live_telemetry(&state, 20),
            "api" | "a" => {
                let port = mesh_port();
                match Runtime::new() {
                    Ok(runtime) => runtime.block_on(serve_http(state.clone(), port)),
                    Err(error) => println!("could not start the runtime: {error}"),
                }
            }
            "reset" => {
                state.with(|session| *session = MeshSession::new());
                println!("mesh reset");
            }
            "help" | "h" => print_menu(),
            "quit" | "q" | "exit" | "e" => break,
            other => println!("unknown command `{other}`; try help"),
        }
    }
}

fn main() {
    let state = AppState::new();
    let arguments: Vec<String> = env::args().skip(1).collect();

    match arguments.first().map(String::as_str) {
        // Non-interactive entry points, so the mesh can be scripted or run as a service.
        Some("serve") | Some("--serve") => {
            let port = mesh_port();
            match Runtime::new() {
                Ok(runtime) => runtime.block_on(serve_http(state, port)),
                Err(error) => eprintln!("could not start the runtime: {error}"),
            }
        }
        Some("scenario") => {
            let name = arguments.get(1).map(String::as_str).unwrap_or("rehearsal");
            command_scenario(&state, name);
            command_overview(&state);
        }
        Some("battle") => {
            let report = AgentBattle::new().run();
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_default()
            );
        }
        Some("--help") | Some("-h") => {
            println!("usage: hatcher-ux [serve | scenario <name> | battle]");
            println!("  no arguments starts the interactive console");
            println!(
                "  HATCHER_MESH_PORT           listen port for `serve` (default {DEFAULT_PORT})"
            );
            println!("  HATCHER_MESH_ALLOWED_ORIGIN CORS origin (default: any)");
            println!("  HATCHER_MESH_BIND_ADDR      listen address (default: 127.0.0.1)");
            println!(
                "  HATCHER_MESH_ALLOW_NON_LOOPBACK_BIND explicit opt-in for container binding"
            );
            println!(
                "  HATCHER_MESH_SHADOW_ONLY    expose only health, contract, and stateless routing"
            );
            println!(
                "  HATCHER_MESH_REQUIRE_INTERNAL_TOKEN require a >=32 character routing token"
            );
        }
        _ => interactive_cli(state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_routing_ranks_only_the_supplied_tenant_cohort() {
        let strong = AgentRegistration::new("strong", "Strong coder", AgentRole::Coder)
            .with_capability(hatcher_core::CapabilityVector::uniform(0.9))
            .with_expertise("rust", 0.95);
        let steady = AgentRegistration::new("steady", "Steady coder", AgentRole::Coder)
            .with_capability(hatcher_core::CapabilityVector::uniform(0.6))
            .with_expertise("rust", 0.60);
        let response = build_shadow_route(ShadowRouteRequest {
            agents: vec![steady, strong],
            task: TaskEnvelope::new("review the Rust adapter").with_domain("rust"),
        })
        .expect("valid cohort should route");

        assert_eq!(response.recommended_agent_id.as_deref(), Some("strong"));
        assert_eq!(response.contract_version, SHADOW_CONTRACT_VERSION);
        assert_eq!(response.candidates.len(), 2);
        assert!(response.confidence >= 0.5);
        assert_eq!(response.domain, "rust");
    }

    #[test]
    fn shadow_routing_rejects_an_empty_cohort() {
        let error = build_shadow_route(ShadowRouteRequest {
            agents: Vec::new(),
            task: TaskEnvelope::new("nothing can run this"),
        })
        .expect_err("an empty tenant must not fall back to another cohort");

        assert_eq!(error.status(), 400);
    }

    #[test]
    fn live_routing_uses_the_versioned_stateless_contract() {
        let response = build_live_route(ShadowRouteRequest {
            agents: vec![AgentRegistration::new(
                "agent-1",
                "Agent one",
                AgentRole::Coder,
            )],
            task: TaskEnvelope::new("route this mission"),
        })
        .expect("a valid live cohort should route");

        assert_eq!(response.contract_version, ROUTING_CONTRACT_VERSION);
        assert_eq!(response.recommended_agent_id.as_deref(), Some("agent-1"));
    }

    #[test]
    fn shadow_routing_caps_the_tenant_cohort() {
        let agents = (0..=MAX_SHADOW_AGENTS)
            .map(|index| {
                AgentRegistration::new(
                    format!("agent-{index}"),
                    format!("Agent {index}"),
                    AgentRole::Coder,
                )
            })
            .collect();
        let error = build_shadow_route(ShadowRouteRequest {
            agents,
            task: TaskEnvelope::new("too many candidates"),
        })
        .expect_err("oversized cohorts must be rejected before ranking");

        assert_eq!(error.status(), 400);
    }

    #[test]
    fn production_token_policy_fails_closed() {
        let token = "0123456789abcdef0123456789abcdef";
        assert!(token_policy_allows(Some(token), true, Some(token)));
        assert!(!token_policy_allows(Some(token), true, None));
        assert!(!token_policy_allows(Some(token), true, Some("wrong")));
        assert!(!token_policy_allows(None, true, Some(token)));
        assert!(token_policy_allows(None, false, None));
    }

    #[test]
    fn non_loopback_binding_requires_explicit_opt_in() {
        assert!(bind_policy_allows(Ipv4Addr::LOCALHOST, false));
        assert!(!bind_policy_allows(Ipv4Addr::UNSPECIFIED, false));
        assert!(bind_policy_allows(Ipv4Addr::UNSPECIFIED, true));
    }

    #[test]
    fn telemetry_render_contains_the_live_numbers() {
        let snapshot = TelemetrySnapshot {
            tick: 4,
            omega: 1.234,
            regime: "compounding".into(),
            agents: 5,
            intelligence: 0.812,
            emergence: 0.25,
            mean_trust: 0.61,
            hubs: 1,
            isolated: 0,
            action: "stabilize".into(),
            verified: true,
        };

        let rendered = snapshot.render();
        assert!(rendered.contains("HatcherLabs Mesh Telemetry"));
        assert!(rendered.contains("compounding"));
        assert!(rendered.contains("stabilize"));
        assert!(rendered.contains("1.234"));
    }

    #[test]
    fn a_telemetry_tick_actually_advances_the_mesh() {
        let state = AppState::new();
        let first = build_telemetry_snapshot(&state, 1);
        let second = build_telemetry_snapshot(&state, 2);

        assert_eq!(state.with(|session| session.adapter.mesh.global.epoch), 2);
        assert_eq!(state.with(|session| session.traces.len()), 2);
        assert_ne!(
            first.omega, second.omega,
            "telemetry must reflect real work"
        );
    }

    #[test]
    fn the_trace_log_stays_bounded() {
        let mut session = MeshSession::new();
        let submission = TaskSubmission {
            description: "fill the log".into(),
            domain: None,
            features: vec![0.5],
            urgency: None,
            execution_mode: None,
        };
        for _ in 0..(TRACE_LOG_CAPACITY + 10) {
            let result = session.submit(&submission);
            session.log(result.trace);
        }
        assert_eq!(session.traces.len(), TRACE_LOG_CAPACITY);
    }

    #[test]
    fn trace_summaries_compress_a_trace_for_list_views() {
        let mut session = MeshSession::new();
        let submission = TaskSubmission {
            description: "summarize me".into(),
            domain: Some("rust".into()),
            features: vec![0.4, 0.6],
            urgency: Some(0.5),
            execution_mode: Some(ExecutionMode::Controlled),
        };
        let result = session.submit(&submission);
        let summary = TraceSummary::from(&result.trace);

        assert_eq!(summary.domain, "rust");
        assert_eq!(summary.digest.len(), 16);
        assert!(!summary.action.is_empty());
    }

    #[test]
    fn state_recovers_from_a_poisoned_lock() {
        let state = AppState::new();
        let panicking = state.clone();
        let _ = std::thread::spawn(move || {
            panicking.with(|_| panic!("boom"));
        })
        .join();

        // A panicked request must not brick the mesh for every later request.
        assert_eq!(state.with(|session| session.adapter.mesh.nodes.len()), 5);
    }

    #[test]
    fn port_defaults_when_the_environment_is_unset() {
        std::env::remove_var("HATCHER_MESH_PORT");
        assert_eq!(mesh_port(), DEFAULT_PORT);
        std::env::set_var("HATCHER_MESH_PORT", "4100");
        assert_eq!(mesh_port(), 4100);
        std::env::set_var("HATCHER_MESH_PORT", "not-a-port");
        assert_eq!(
            mesh_port(),
            DEFAULT_PORT,
            "garbage falls back instead of panicking"
        );
        std::env::remove_var("HATCHER_MESH_PORT");
    }
}
