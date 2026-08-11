//! # HatcherLabs Agent Mesh Neural System — observatory
//!
//! `hatcher-terminal` is the visual surface of HAMNS. It links the whole mesh —
//! [`hatcher_core`] contracts, the [`hatcher_neural`] engine, and the
//! [`hatcher_playground`] arena — and renders a running mesh as 3D ANSI in the
//! terminal, so the behaviour described by the ten equations can be watched
//! rather than inferred from logs.
//!
//! ## What it shows
//!
//! | View | What it answers |
//! |---|---|
//! | Active node tornado | Is the mesh live, and which agents are carrying it? |
//! | Node graph | Who is connected to whom, and how strongly? |
//! | Trust matrix | Where is `T_ij` concentrating? |
//! | Equations | What are all ten update rules doing right now? |
//! | Pipeline | Where did the last task actually go, stage by stage? |
//! | Contract | What is driving the mesh, and is any of it real? |
//! | Benchmark | Does mesh routing beat a fixed assignment? |
//! | Economics | What does this cohort cost, and how fast is it? |
//!
//! The last three exist because once something outside the mesh starts driving it,
//! "the tornado is spinning" stops being the interesting question. A mesh running on
//! declared numbers looks identical to one running on measured ones, so the contract
//! tab says which it is and the economics tab marks every unmeasured figure as a
//! declaration.
//!
//! The tornado is the headline. Radius is inverse capability, height is trust
//! standing, angular speed is live activation, and the funnel's amplitude tracks
//! `Ω` — so a mesh that is genuinely working spins up a tall, fast, bright
//! vortex, and one that is stalling flattens into a slow dim ring.
//!
//! ## Minimal use
//!
//! ```rust
//! use hatcher_core::TaskSpec;
//! use hatcher_neural::{pipeline, NeuralMesh};
//! use hatcher_terminal::{Canvas, Camera, Observation, View, dashboard};
//!
//! let mut mesh = NeuralMesh::default();
//! let features = vec![0.3, 0.7, 0.2, 0.9];
//! let task = TaskSpec::new("task-1", "ship the observatory", "rust")
//!     .with_features(features.clone())
//!     .parsed_from_features();
//!
//! let trace = pipeline::run(&mut mesh, &task);
//! let observation = Observation::capture(&mesh, &features, true, Some(trace));
//!
//! let mut canvas = Canvas::new(120, 34);
//! dashboard::draw(&mut canvas, View::Dashboard, &observation, &Camera::default(), 0.0, 0);
//! print!("{}", canvas.render());
//! ```
//!
//! Everything renders through [`Canvas`], which is depth-buffered and emits
//! truecolor ANSI — or plain text via [`Canvas::render_plain`], which is what
//! makes the whole view layer testable without a terminal.

pub mod canvas;
pub mod dashboard;
pub mod observe;
pub mod panels;
pub mod space;
pub mod theme;
pub mod tornado;

pub use canvas::{Canvas, Viewport};
pub use dashboard::View;
pub use observe::{AgentView, Observation};
pub use panels::Rect;
pub use space::{Camera, Vec3};
pub use theme::Rgb;
pub use tornado::{Tornado, VortexNode};

/// The version of the mesh this observatory was built against.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::TaskSpec;
    use hatcher_neural::{pipeline, NeuralMesh};

    #[test]
    fn the_observatory_renders_a_live_mesh_end_to_end() {
        let mut mesh = NeuralMesh::default();
        let features = vec![0.3, 0.7, 0.2, 0.9];
        let task = TaskSpec::new("task-1", "ship it", "rust")
            .with_features(features.clone())
            .parsed_from_features();

        let trace = pipeline::run(&mut mesh, &task);
        assert_eq!(trace.stages.len(), 10);

        let observation = Observation::capture(&mesh, &features, true, Some(trace));
        let mut canvas = Canvas::new(120, 34);
        dashboard::draw(&mut canvas, View::Dashboard, &observation, &Camera::default(), 0.0, 0);

        let ansi = canvas.render();
        assert!(ansi.contains("\x1b[38;2;"), "the observatory renders truecolor");
        assert!(canvas.render_plain().contains("MESH ACTIVE"));
    }

    #[test]
    fn version_is_the_release_under_test() {
        assert_eq!(VERSION, "1.0.0");
    }
}
