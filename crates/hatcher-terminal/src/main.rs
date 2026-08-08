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

use hatcher_core::TaskSpec;
use hatcher_neural::{pipeline, NeuralMesh};
use hatcher_playground::Scenario;
use hatcher_terminal::{dashboard, Camera, Canvas, Observation, View, VERSION};

const ALT_SCREEN_ENTER: &str = "\x1b[?1049h\x1b[?25l";
const ALT_SCREEN_LEAVE: &str = "\x1b[?25h\x1b[?1049l";
const HOME: &str = "\x1b[H";

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
        }
    }
}

fn usage() -> String {
    format!(
        "hatcher-terminal {VERSION} — the HAMNS observatory

USAGE:
    hatcher-terminal [OPTIONS]

OPTIONS:
    --view <name>       dashboard | tornado | graph | equations   [default: dashboard]
    --scenario <name>   rehearsal | stress | frontier             [default: rehearsal]
    --size <WxH>        canvas size in cells                      [default: 120x34]
    --frames <n>        stop after n frames (0 = run forever)     [default: 0]
    --fps <n>           target frames per second                  [default: 24]
    --spin <f>          camera orbit speed, radians/second        [default: 0.55]
    --once              render a single frame to stdout and exit
    --plain             no colour, no alternate screen
    -h, --help          show this help

The mesh is live: each frame runs real work through the ten-stage pipeline and
renders the resulting state. An idle mesh looks idle — the tornado only spins up
when the mesh is actually executing."
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
                options.width = w.trim().parse::<usize>().map_err(|e| e.to_string())?.clamp(20, 400);
                options.height = h.trim().parse::<usize>().map_err(|e| e.to_string())?.clamp(8, 200);
                index += 2;
            }
            "--frames" => {
                options.frames = value()?.parse::<u64>().map_err(|e| e.to_string())?;
                index += 2;
            }
            "--fps" => {
                options.fps = value()?.parse::<u64>().map_err(|e| e.to_string())?.clamp(1, 120);
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

fn run(options: Options) -> io::Result<()> {
    let mut mesh = NeuralMesh::default();
    let batch: Vec<TaskSpec> = options.scenario.task_batch();
    let mut canvas = Canvas::new(options.width, options.height);
    let mut camera = Camera::default();

    let stdout = io::stdout();
    let mut out = stdout.lock();

    let animated = !options.once;
    if animated && !options.plain {
        write!(out, "{ALT_SCREEN_ENTER}")?;
    }

    let start = Instant::now();
    let interval = Duration::from_millis(1000 / options.fps.max(1));
    let mut frame: u64 = 0;
    let mut last_trace;

    loop {
        // Feed the mesh real work: one task per frame, cycled through the batch.
        // This is what makes "active" mean something — the tornado is reacting to
        // the pipeline, not to a timer.
        let task = &batch[(frame as usize) % batch.len().max(1)];
        last_trace = Some(pipeline::run(&mut mesh, task));
        let active = true;

        let time = start.elapsed().as_secs_f64();
        camera.yaw = time * options.spin;

        let observation = Observation::capture(&mesh, &task.features, active, last_trace.clone());
        dashboard::draw(&mut canvas, options.view, &observation, &camera, time, frame);

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
            "--view", "tornado", "--size", "80x20", "--frames", "3", "--fps", "10", "--once", "--plain",
            "--scenario", "stress", "--spin", "1.25",
        ]))
        .expect("valid flags");

        assert_eq!(options.view, View::Tornado);
        assert_eq!((options.width, options.height), (80, 20));
        assert_eq!(options.frames, 3);
        assert_eq!(options.fps, 10);
        assert!(options.once && options.plain);
        assert_eq!(options.scenario.name, "stress");
        assert!((options.spin - 1.25).abs() < 1e-12);
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
