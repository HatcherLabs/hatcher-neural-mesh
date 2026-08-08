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
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use hatcher_core::{
    ApiRequest, ApiResponse, ExecutionMode, MeshCoefficients, PipelineTrace, TaskSubmission,
};
use hatcher_playground::{AgentArena, AgentBattle, Scenario};
use serde::Serialize;
use serde_json::json;
use tokio::runtime::Runtime;
use warp::http::StatusCode;
use warp::Filter;

/// How many pipeline traces the Logs view keeps.
const TRACE_LOG_CAPACITY: usize = 200;

/// Default port for the mesh sidecar.
const DEFAULT_PORT: u16 = 3030;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<MeshSession>>,
}

struct MeshSession {
    arena: AgentArena,
    traces: Vec<PipelineTrace>,
    execution_mode: ExecutionMode,
}

impl MeshSession {
    fn new() -> Self {
        Self {
            arena: AgentArena::new().with_policy_from_env(),
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

fn with_state(state: AppState) -> impl Filter<Extract = (AppState,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}

fn routes(state: AppState) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    let health = warp::path!("health")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            let (omega, agents, epoch) = state.with(|session| {
                (
                    session.arena.mesh.global.omega,
                    session.arena.mesh.nodes.len(),
                    session.arena.mesh.global.epoch,
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
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.arena.mesh.overview())));

    let analytics = warp::path!("api" / "mesh" / "analytics")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.arena.mesh.analytics())));

    let graph = warp::path!("api" / "mesh" / "graph")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.arena.mesh.graph_view())));

    let trust = warp::path!("api" / "mesh" / "trust")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.arena.mesh.trust_view())));

    let agents = warp::path!("api" / "mesh" / "agents")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.arena.mesh.agent_summaries())));

    let get_config = warp::path!("api" / "mesh" / "config")
        .and(warp::get())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            warp::reply::json(&state.with(|s| {
                let mode = s.execution_mode;
                s.arena.mesh.config_view(mode)
            }))
        });

    // Coefficients are validated before they are installed: a bad tuning would make
    // every subsequent equation diverge, and a 400 is far easier to debug than a mesh
    // that quietly stopped converging.
    let put_config = warp::path!("api" / "mesh" / "config")
        .and(warp::put().or(warp::post()).unify())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(|coefficients: MeshCoefficients, state: AppState| -> Box<dyn warp::Reply> {
            match coefficients.validate() {
                Ok(()) => {
                    let view = state.with(|session| {
                        session.arena.mesh.coefficients = coefficients;
                        let mode = session.execution_mode;
                        session.arena.mesh.config_view(mode)
                    });
                    Box::new(warp::reply::json(&view))
                }
                Err(error) => Box::new(warp::reply::with_status(
                    warp::reply::json(&json!({ "error": error.to_string() })),
                    StatusCode::BAD_REQUEST,
                )),
            }
        });

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
        .map(|state: AppState| warp::reply::json(&state.with(|s| s.arena.mesh.memory.clone())));

    let submit_task = warp::path!("api" / "tasks")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .map(|submission: TaskSubmission, state: AppState| {
            let result = state.with(|session| {
                let result = session.arena.submit(&submission);
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
                    warp::reply::json(&json!({ "error": format!("no trace for task `{task_id}`") })),
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
                let hatcher = session.arena.evaluate(&request.agent_id, &request.prompt, request.features.clone());
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
                session
                    .arena
                    .run_simulation(&request.agent_id, &request.prompt, request.features.clone(), 5)
            });
            warp::reply::json(&value)
        });

    let battle = warp::path!("api" / "battle")
        .and(warp::get())
        .map(|| warp::reply::json(&AgentBattle::new().run()));

    let reset = warp::path!("api" / "mesh" / "reset")
        .and(warp::post())
        .and(with_state(state.clone()))
        .map(|state: AppState| {
            let overview = state.with(|session| {
                *session = MeshSession::new();
                session.arena.mesh.overview()
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

    health
        .or(overview)
        .or(analytics)
        .or(graph)
        .or(trust)
        .or(agents)
        .or(get_config)
        .or(put_config)
        .or(logs)
        .or(memory)
        .or(submit_task)
        .or(list_tasks)
        .or(trace_by_id)
        .or(infer)
        .or(simulate)
        .or(battle)
        .or(reset)
        .with(cors)
}

fn mesh_port() -> u16 {
    env::var("HATCHER_MESH_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

async fn serve_http(state: AppState, port: u16) {
    println!("mesh API listening on http://127.0.0.1:{port}");
    println!("  GET  /health");
    println!("  GET  /api/mesh/overview | analytics | graph | trust | agents | logs | memory");
    println!("  GET  /api/mesh/config     PUT /api/mesh/config");
    println!("  POST /api/tasks           GET /api/tasks   GET /api/tasks/{{id}}");
    println!("  POST /api/infer | /api/simulate | /api/mesh/reset");
    println!("  GET  /api/battle");
    warp::serve(routes(state)).run(([127, 0, 0, 1], port)).await;
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
        let result = session.arena.submit(&submission);
        session.log(result.trace.clone());

        let overview = session.arena.mesh.overview();
        TelemetrySnapshot {
            tick,
            omega: overview.omega,
            regime: overview.regime.as_str().to_string(),
            agents: overview.agent_count,
            intelligence: overview.intelligence.total,
            emergence: overview.emergence_ratio,
            mean_trust: session.arena.mesh.trust.mean_trust(),
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
        let result = session.arena.submit(&submission);
        session.log(result.trace.clone());
        result.trace
    });

    println!("task {} | {} | P={:.3} ({})", trace.task_id, trace.domain, trace.priority.value, trace.priority.band.as_str());
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
        trace.decision.action, trace.decision.confidence, trace.verified, trace.omega_before, trace.omega_after
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
        let (report, traces) = hatcher_playground::run_scenario(&mut session.arena.mesh, &scenario);
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
    let overview = state.with(|session| session.arena.mesh.overview());
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
        "{:<14} {:<12} {:>6} {:>6} {:>6} {:>6} {:>5} {:>4}  {}",
        "agent", "role", "A_i", "conf", "in_T", "R_i", "deg", "att", "bottleneck"
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
        (session.arena.mesh.trust_view(), session.arena.mesh.top_collaborations(5))
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
    let analytics = state.with(|session| session.arena.mesh.analytics());
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
    println!("{:>6} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}", "epoch", "omega", "L", "E", "C", "F", "D");
    for sample in analytics.series.iter().rev().take(15).collect::<Vec<_>>().into_iter().rev() {
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
    println!("{:<18} {:<10} {:<10} {:<9} {:>7} {:>8}", "task", "action", "band", "verified", "P", "omega");
    for row in rows {
        println!(
            "{:<18} {:<10} {:<10} {:<9} {:>7.3} {:>8.4}",
            row.task_id, row.action, row.band, row.verified, row.priority, row.omega_after
        );
    }
}

fn command_memory(state: &AppState) {
    let memory = state.with(|session| session.arena.mesh.memory.clone());
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
            println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
        }
        Some("--help") | Some("-h") => {
            println!("usage: hatcher-ux [serve | scenario <name> | battle]");
            println!("  no arguments starts the interactive console");
            println!("  HATCHER_MESH_PORT           listen port for `serve` (default {DEFAULT_PORT})");
            println!("  HATCHER_MESH_ALLOWED_ORIGIN CORS origin (default: any)");
        }
        _ => interactive_cli(state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert_eq!(state.with(|session| session.arena.mesh.global.epoch), 2);
        assert_eq!(state.with(|session| session.traces.len()), 2);
        assert_ne!(first.omega, second.omega, "telemetry must reflect real work");
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
            let result = session.arena.submit(&submission);
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
        let result = session.arena.submit(&submission);
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
        assert_eq!(state.with(|session| session.arena.mesh.nodes.len()), 5);
    }

    #[test]
    fn port_defaults_when_the_environment_is_unset() {
        std::env::remove_var("HATCHER_MESH_PORT");
        assert_eq!(mesh_port(), DEFAULT_PORT);
        std::env::set_var("HATCHER_MESH_PORT", "4100");
        assert_eq!(mesh_port(), 4100);
        std::env::set_var("HATCHER_MESH_PORT", "not-a-port");
        assert_eq!(mesh_port(), DEFAULT_PORT, "garbage falls back instead of panicking");
        std::env::remove_var("HATCHER_MESH_PORT");
    }
}
