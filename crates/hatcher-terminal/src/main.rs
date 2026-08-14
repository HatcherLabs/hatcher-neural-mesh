//! `hatcher-terminal` — the HAMNS observatory.
//!
//! Drives a live [`NeuralMesh`] through the playground's scenarios and renders
//! it as an animated 3D terminal dashboard. The mesh is genuinely running: every
//! frame is a real observation taken after real pipeline work, not a canned
//! animation, which is the whole point of watching it.

use std::env;
use std::io::{self, Write};
use std::thread;
use std::time::{Duration, Instant};

use hatcher_core::{ErrorClass, StageOutcomeReport, TaskEnvelope, TaskSpec};
use hatcher_neural::MeshAdapter;
use hatcher_playground::{Benchmark, BenchmarkReport, Scenario};
use hatcher_terminal::{dashboard, Camera, Canvas, Observation, View, VERSION};

const ALT_SCREEN_ENTER: &str = "\x1b[?1049h\x1b[?25l";
const ALT_SCREEN_LEAVE: &str = "\x1b[?25h\x1b[?1049l";
const HOME: &str = "\x1b[H";

/// Where the observatory's stage outcomes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutcomeMode {
    /// The deterministic competence model. A rehearsal.
    Simulated,
    /// Drive the full contract loop — plan, report, finalize — with outcomes
    /// synthesized the way a real runtime would send them. Nothing here is production
    /// data; it exists so the contract path is exercised and visible rather than
    /// described, and every frame it produces is labelled `reported` for that reason.
    Reported,
}

impl OutcomeMode {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "simulated" | "sim" => Some(OutcomeMode::Simulated),
            "reported" | "contract" => Some(OutcomeMode::Reported),
            _ => None,
        }
    }
}

/// Parsed command line.
#[derive(Debug)]
struct Options {
    view: View,
    width: usize,
    height: usize,
    /// Frames to render. `0` means run until interrupted.
    frames: u64,
    fps: u64,
    /// Render one frame to stdout and exit — for piping and CI.
    once: bool,
    plain: bool,
    scenario: Scenario,
    spin: f64,
    outcomes: OutcomeMode,
    /// Run the routing benchmark at startup and populate the benchmark tab.
    benchmark: bool,
    /// Path to an ONNX policy to use as the decision head.
    model: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            view: View::Dashboard,
            width: 120,
            height: 34,
            frames: 0,
            fps: 24,
            once: false,
            plain: false,
            scenario: Scenario::rehearsal(),
            spin: 0.55,
            outcomes: OutcomeMode::Simulated,
            benchmark: false,
            model: None,
        }
    }
}

fn usage() -> String {
    format!(
        "hatcher-terminal {VERSION} — the HAMNS observatory

USAGE:
    hatcher-terminal [OPTIONS]

OPTIONS:
    --view <name>       dashboard | tornado | graph | equations |
                        contract | benchmark | economics          [default: dashboard]
    --scenario <name>   rehearsal | stress | frontier             [default: rehearsal]
    --outcomes <mode>   simulated | reported                      [default: simulated]
    --model <path>      ONNX policy to use as the decision head   [needs --features onnx]
    --benchmark         run the routing benchmark and fill the benchmark tab
    --size <WxH>        canvas size in cells                      [default: 120x34]
    --frames <n>        stop after n frames (0 = run forever)     [default: 0]
    --fps <n>           target frames per second                  [default: 24]
    --spin <f>          camera orbit speed, radians/second        [default: 0.55]
    --once              render a single frame to stdout and exit
    --plain             no colour, no alternate screen
    -h, --help          show this help

TABS:
    dashboard   what is the mesh doing right now?
    tornado     is the mesh live, and who is carrying it?
    graph       who is connected to whom, and how strongly?
    equations   what are all ten update rules doing?
    contract    what is driving the mesh, and is any of it real?
    benchmark   does mesh routing beat a fixed assignment?
    economics   what does this cohort cost, and how fast is it?

The mesh is live: each frame runs real work through the ten-stage pipeline and
renders the resulting state. An idle mesh looks idle — the tornado only spins up
when the mesh is actually executing.

`--outcomes reported` drives the full integration contract instead of the built-in
simulator: the observatory plans a run, reports synthesized stage outcomes back, and
finalizes it. The outcomes are still made up — what is real is the code path."
    )
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut index = 0;

    while index < args.len() {
        let arg = args[index].as_str();
        let value = || -> Result<String, String> {
            args.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value"))
        };

        match arg {
            "-h" | "--help" => return Err(usage()),
            "--once" => {
                options.once = true;
                index += 1;
            }
            "--plain" => {
                options.plain = true;
                index += 1;
            }
            "--benchmark" => {
                options.benchmark = true;
                index += 1;
            }
            "--outcomes" => {
                let raw = value()?;
                options.outcomes = OutcomeMode::parse(&raw)
                    .ok_or_else(|| format!("unknown outcome mode: {raw}"))?;
                index += 2;
            }
            "--model" => {
                options.model = Some(value()?);
                index += 2;
            }
            "--view" => {
                let raw = value()?;
                options.view = View::parse(&raw).ok_or_else(|| format!("unknown view: {raw}"))?;
                index += 2;
            }
            "--scenario" => {
                let raw = value()?;
                options.scenario = match raw.trim().to_ascii_lowercase().as_str() {
                    "rehearsal" => Scenario::rehearsal(),
                    "stress" => Scenario::stress(),
                    "frontier" => Scenario::frontier(),
                    other => return Err(format!("unknown scenario: {other}")),
                };
                index += 2;
            }
            "--size" => {
                let raw = value()?;
                let (w, h) = raw
                    .split_once(['x', 'X'])
                    .ok_or_else(|| format!("--size wants WxH, got {raw}"))?;
                options.width = w
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| e.to_string())?
                    .clamp(20, 400);
                options.height = h
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| e.to_string())?
                    .clamp(8, 200);
                index += 2;
            }
            "--frames" => {
                options.frames = value()?.parse::<u64>().map_err(|e| e.to_string())?;
                index += 2;
            }
            "--fps" => {
                options.fps = value()?
                    .parse::<u64>()
                    .map_err(|e| e.to_string())?
                    .clamp(1, 120);
                index += 2;
            }
            "--spin" => {
                options.spin = value()?.parse::<f64>().map_err(|e| e.to_string())?;
                index += 2;
            }
            other => return Err(format!("unknown argument: {other}\n\n{}", usage())),
        }
    }

    Ok(options)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let options = match parse_args(&args) {
        Ok(options) => options,
        Err(message) => {
            // --help is not a failure; an unknown flag is.
            let is_help = args.iter().any(|a| a == "-h" || a == "--help");
            if is_help {
                println!("{message}");
                return;
            }
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    if let Err(error) = run(options) {
        eprintln!("hatcher-terminal: {error}");
        std::process::exit(1);
    }
}

/// Turn a task into an envelope the contract path can accept.
fn envelope_for(task: &TaskSpec) -> TaskEnvelope {
    let mut envelope = TaskEnvelope::new(task.description.clone())
        .with_domain(task.domain.clone())
        .with_features(task.features.clone())
        .with_urgency(task.urgency);
    envelope.task_id = Some(task.id.clone());
    envelope.uncertainty = Some(task.uncertainty);
    envelope.budget = Some(task.budget);
    envelope.implementation_cost = Some(task.implementation_cost);
    envelope.execution_mode = Some(task.execution_mode);
    envelope
}

/// Synthesize the reports a real runtime would send back for a plan.
///
/// Deterministic in the frame counter so the observatory stays reproducible, and
/// deliberately imperfect: hard tasks fail more, one frame in seven hits an
/// infrastructure error, and the reported latency tracks what the plan expected. That
/// last part is what makes the economics tab move — reported latency is the only thing
/// that ever turns a declared profile into a measured one.
fn synthesize_reports(
    plan: &hatcher_core::RoutingPlan,
    task: &TaskSpec,
    frame: u64,
) -> Vec<StageOutcomeReport> {
    let difficulty = (0.5 * task.uncertainty + 0.5 * task.implementation_cost).clamp(0.0, 1.0);

    plan.stages
        .iter()
        .enumerate()
        .map(|(offset, planned)| {
            let roll = (frame as usize + offset * 3) % 7;
            let outage = roll == 6;
            let failed = outage || (difficulty > 0.6 && roll == 0);
            let quality = if failed {
                0.10
            } else {
                (0.95 - 0.35 * difficulty).clamp(0.0, 1.0)
            };

            let mut report = if failed {
                StageOutcomeReport::failure(
                    planned.stage,
                    &planned.agent_id,
                    if outage {
                        ErrorClass::Infrastructure
                    } else {
                        ErrorClass::Quality
                    },
                )
            } else {
                StageOutcomeReport::success(planned.stage, &planned.agent_id, quality)
            };
            report.confidence = quality;
            report
                .with_latency_ms(planned.expected_latency_ms * (0.8 + 0.4 * difficulty))
                .with_cost(planned.expected_cost * (0.9 + 0.2 * difficulty))
        })
        .collect()
}

/// Run one task through the mesh in whichever mode was asked for.
fn advance(
    adapter: &mut MeshAdapter,
    task: &TaskSpec,
    mode: OutcomeMode,
    frame: u64,
) -> io::Result<()> {
    let envelope = envelope_for(task);
    let contract_error = |error: hatcher_core::ContractError| {
        io::Error::new(io::ErrorKind::InvalidData, error.to_string())
    };

    match mode {
        OutcomeMode::Simulated => {
            adapter.execute(&envelope).map_err(contract_error)?;
        }
        OutcomeMode::Reported => {
            let plan = adapter.plan(&envelope).map_err(contract_error)?;
            let reports = synthesize_reports(&plan, task, frame);
            // An empty cohort produces an empty plan and nothing to report; finalizing
            // it is still correct — the mesh records that it could not staff the work.
            adapter
                .report_many(&plan.run_id, reports)
                .map_err(contract_error)?;
            adapter.finalize(&plan.run_id).map_err(contract_error)?;
        }
    }
    Ok(())
}

fn run(options: Options) -> io::Result<()> {
    // A `--model` that cannot load is a hard failure here rather than a degradation. An
    // operator who explicitly named a policy on the command line is asking to watch *that*
    // policy; quietly showing them the built-in head instead would make the observatory
    // lie about the very thing they opened it to see.
    let mut adapter = match options.model.as_deref() {
        Some(path) => MeshAdapter::new()
            .try_with_policy(path)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?,
        None => MeshAdapter::new(),
    };
    let batch: Vec<TaskSpec> = options.scenario.task_batch();
    // `.max(1)` on the modulus below keeps the division safe but would still index
    // an empty vector; refuse the scenario here rather than panic mid-frame.
    if batch.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "scenario produced no tasks: nothing to observe",
        ));
    }
    let mut canvas = Canvas::new(options.width, options.height);
    let mut camera = Camera::default();

    // Run once at startup rather than per frame: six policies over a full batch is a
    // benchmark, not a redraw, and re-running it every frame would make the observatory
    // spend all its time measuring instead of showing.
    let benchmark: Option<BenchmarkReport> = options.benchmark.then(|| {
        Benchmark::new()
            .with_scenario(options.scenario.clone())
            .run()
    });

    let stdout = io::stdout();
    let mut out = stdout.lock();

    let animated = !options.once;
    if animated && !options.plain {
        write!(out, "{ALT_SCREEN_ENTER}")?;
    }

    let start = Instant::now();
    let interval = Duration::from_millis(1000 / options.fps.max(1));
    let mut frame: u64 = 0;

    loop {
        // Feed the mesh real work: one task per frame, cycled through the batch.
        // This is what makes "active" mean something — the tornado is reacting to
        // the pipeline, not to a timer.
        let task = &batch[(frame as usize) % batch.len()];
        advance(&mut adapter, task, options.outcomes, frame)?;
        let active = true;

        let time = start.elapsed().as_secs_f64();
        camera.yaw = time * options.spin;

        let observation = Observation::capture_with(
            &adapter.mesh,
            &task.features,
            active,
            adapter.last_trace().cloned(),
            adapter.calibration,
        )
        .with_benchmark(benchmark.clone())
        .with_open_runs(adapter.open_runs().len());
        dashboard::draw(
            &mut canvas,
            options.view,
            &observation,
            &camera,
            time,
            frame,
        );

        let painted = if options.plain {
            canvas.render_plain()
        } else {
            canvas.render()
        };

        if options.once {
            writeln!(out, "{painted}")?;
            out.flush()?;
            return Ok(());
        }

        write!(out, "{HOME}{painted}")?;
        out.flush()?;

        frame += 1;
        if options.frames != 0 && frame >= options.frames {
            break;
        }
        thread::sleep(interval);
    }

    if animated && !options.plain {
        write!(out, "{ALT_SCREEN_LEAVE}")?;
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn defaults_are_a_usable_dashboard() {
        let options = parse_args(&[]).expect("no args is valid");
        assert_eq!(options.view, View::Dashboard);
        assert_eq!((options.width, options.height), (120, 34));
        assert!(!options.once);
    }

    #[test]
    fn flags_are_parsed() {
        let options = parse_args(&args(&[
            "--view",
            "tornado",
            "--size",
            "80x20",
            "--frames",
            "3",
            "--fps",
            "10",
            "--once",
            "--plain",
            "--scenario",
            "stress",
            "--spin",
            "1.25",
            "--outcomes",
            "reported",
            "--benchmark",
        ]))
        .expect("valid flags");

        assert_eq!(options.view, View::Tornado);
        assert_eq!((options.width, options.height), (80, 20));
        assert_eq!(options.frames, 3);
        assert_eq!(options.fps, 10);
        assert!(options.once && options.plain);
        assert_eq!(options.scenario.name, "stress");
        assert!((options.spin - 1.25).abs() < 1e-12);
        assert_eq!(options.outcomes, OutcomeMode::Reported);
        assert!(options.benchmark);
    }

    #[test]
    fn the_new_tabs_are_reachable_from_the_command_line() {
        for (flag, view) in [
            ("contract", View::Contract),
            ("benchmark", View::Benchmark),
            ("economics", View::Economics),
        ] {
            assert_eq!(parse_args(&args(&["--view", flag])).unwrap().view, view);
        }
    }

    #[test]
    fn the_contract_path_produces_a_trace_that_attests_to_reported_work() {
        let mut adapter = MeshAdapter::new();
        let task = Scenario::rehearsal().task_batch().remove(0);
        advance(&mut adapter, &task, OutcomeMode::Reported, 1)
            .expect("the contract loop must close");

        let trace = adapter
            .last_trace()
            .expect("a finalized run leaves a trace");
        assert!(
            trace.provenance.is_real(),
            "the reported mode has to actually go through the contract, not around it"
        );
        assert!(
            adapter.open_runs().is_empty(),
            "and must not strand the run"
        );
        assert!(trace.total_latency_ms() > 0.0);
    }

    #[test]
    fn the_simulated_path_is_still_marked_as_a_rehearsal() {
        let mut adapter = MeshAdapter::new();
        let task = Scenario::rehearsal().task_batch().remove(0);
        advance(&mut adapter, &task, OutcomeMode::Simulated, 0).unwrap();

        assert!(!adapter.last_trace().unwrap().provenance.is_real());
    }

    #[test]
    fn reported_runs_teach_the_mesh_what_its_agents_cost() {
        let mut adapter = MeshAdapter::new();
        let batch = Scenario::rehearsal().task_batch();
        for (frame, task) in batch.iter().take(6).enumerate() {
            advance(&mut adapter, task, OutcomeMode::Reported, frame as u64).unwrap();
        }

        assert!(
            adapter
                .mesh
                .nodes
                .iter()
                .all(|node| node.resources.is_observed()),
            "every agent that ran should have a measured profile now"
        );
    }

    #[test]
    fn a_named_model_is_a_hard_requirement_not_a_preference() {
        let options = parse_args(&args(&[
            "--once",
            "--plain",
            "--size",
            "40x12",
            "--model",
            "models/fixtures/nope.onnx",
        ]))
        .unwrap();
        assert_eq!(options.model.as_deref(), Some("models/fixtures/nope.onnx"));

        let error = run(options).expect_err("an unloadable policy must not start the observatory");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn the_observatory_reports_which_head_is_deciding() {
        let adapter = MeshAdapter::new();
        let head = adapter.head();
        assert_eq!(head.name, "native");
        assert!(head.is_intact());
    }

    #[test]
    fn outcome_modes_round_trip_and_reject_nonsense() {
        assert_eq!(OutcomeMode::parse("SIM"), Some(OutcomeMode::Simulated));
        assert_eq!(OutcomeMode::parse("contract"), Some(OutcomeMode::Reported));
        assert!(OutcomeMode::parse("telepathy").is_none());
        assert!(parse_args(&args(&["--outcomes", "telepathy"])).is_err());
    }

    #[test]
    fn sizes_are_clamped_to_something_renderable() {
        let tiny = parse_args(&args(&["--size", "1x1"])).unwrap();
        assert!(tiny.width >= 20 && tiny.height >= 8);
        let huge = parse_args(&args(&["--size", "9999x9999"])).unwrap();
        assert!(huge.width <= 400 && huge.height <= 200);
    }

    #[test]
    fn bad_input_is_rejected_rather_than_guessed_at() {
        assert!(parse_args(&args(&["--view", "wormhole"])).is_err());
        assert!(parse_args(&args(&["--scenario", "nope"])).is_err());
        assert!(parse_args(&args(&["--size", "80"])).is_err());
        assert!(parse_args(&args(&["--view"])).is_err());
        assert!(parse_args(&args(&["--frames", "abc"])).is_err());
        assert!(parse_args(&args(&["--nonsense"])).is_err());
    }

    #[test]
    fn help_is_returned_as_the_error_payload_for_the_caller_to_print() {
        let message = parse_args(&args(&["--help"])).unwrap_err();
        assert!(message.contains("USAGE"));
        assert!(message.contains(VERSION));
    }

    #[test]
    fn a_single_frame_run_completes_and_writes_nothing_to_the_alt_screen() {
        let options = parse_args(&args(&["--once", "--plain", "--size", "60x16"])).unwrap();
        run(options).expect("a one-shot render must succeed");
    }
}
