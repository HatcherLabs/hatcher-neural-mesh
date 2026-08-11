//! Layout: composing the panels into one frame.

use crate::canvas::{Canvas, Viewport};
use crate::observe::Observation;
use crate::panels::{self, Rect};
use crate::space::Camera;
use crate::theme;

/// Which view the observatory is showing.
///
/// The first four answer "what is the mesh doing"; the last three answer "is it
/// integration-ready, does its routing pay for itself, and what does it cost" — the
/// questions that matter once something outside the mesh starts driving it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// Everything at once.
    Dashboard,
    /// The tornado, full-screen.
    Tornado,
    /// The 3D node graph, full-screen.
    Graph,
    /// Equations and pipeline side by side.
    Equations,
    /// The integration contract, and the last run's stage outcomes.
    Contract,
    /// Mesh routing against the baselines.
    Benchmark,
    /// What the cohort costs and how fast it is.
    Economics,
}

impl View {
    /// Every view, in tab order.
    pub const ALL: [View; 7] = [
        View::Dashboard,
        View::Tornado,
        View::Graph,
        View::Equations,
        View::Contract,
        View::Benchmark,
        View::Economics,
    ];

    pub fn parse(value: &str) -> Option<View> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dashboard" | "all" => Some(View::Dashboard),
            "tornado" | "vortex" => Some(View::Tornado),
            "graph" | "nodes" => Some(View::Graph),
            "equations" | "eq" => Some(View::Equations),
            "contract" | "adapter" | "integration" => Some(View::Contract),
            "benchmark" | "bench" => Some(View::Benchmark),
            "economics" | "cost" | "econ" => Some(View::Economics),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            View::Dashboard => "dashboard",
            View::Tornado => "tornado",
            View::Graph => "graph",
            View::Equations => "equations",
            View::Contract => "contract",
            View::Benchmark => "benchmark",
            View::Economics => "economics",
        }
    }

    /// The question this view answers.
    pub fn question(&self) -> &'static str {
        match self {
            View::Dashboard => "what is the mesh doing right now?",
            View::Tornado => "is the mesh live, and who is carrying it?",
            View::Graph => "who is connected to whom, and how strongly?",
            View::Equations => "what are all ten update rules doing?",
            View::Contract => "what is driving the mesh, and is any of it real?",
            View::Benchmark => "does mesh routing beat a fixed assignment?",
            View::Economics => "what does this cohort cost, and how fast is it?",
        }
    }
}

/// Draw one full frame.
///
/// `time` drives animation and `frame` is the counter shown in the header; both
/// are passed in rather than read from a clock so a frame is reproducible.
pub fn draw(canvas: &mut Canvas, view: View, observation: &Observation, camera: &Camera, time: f64, frame: u64) {
    canvas.clear();
    let width = canvas.width();
    let height = canvas.height();
    panels::header(canvas, width, observation, frame);

    // The tab strip only earns its row when there is a body left to label. On a very
    // short terminal the panels matter more than knowing what else exists.
    let tabbed = height >= 8;
    if tabbed {
        panels::tabs(canvas, width, view, 1);
    }

    let body_y = if tabbed { 2i64 } else { 1i64 };
    let body_h = height.saturating_sub(if tabbed { 3 } else { 2 });
    if body_h < 3 || width < 24 {
        // Too small to lay anything out; the header alone is the honest answer.
        return;
    }

    match view {
        View::Tornado => {
            let rect = Rect::new(0, body_y, width, body_h);
            canvas.frame(rect.x, rect.y, rect.w, rect.h, "ACTIVE NODE TORNADO", theme::FRAME);
            let inner = rect.inner();
            let clip = canvas.set_clip(inner.x, inner.y, inner.w, inner.h);
            observation.tornado().draw_fitted(
                canvas,
                camera,
                Viewport::new(rect.center_x(), rect.center_y(), inner.w, inner.h),
                time,
            );
            canvas.restore_clip(clip);
        }
        View::Graph => {
            panels::node_graph(canvas, Rect::new(0, body_y, width, body_h), observation, camera);
        }
        View::Equations => {
            let half = width / 2;
            panels::equations(canvas, Rect::new(0, body_y, half, body_h), observation);
            panels::pipeline(
                canvas,
                Rect::new(half as i64, body_y, width - half, body_h),
                observation,
            );
        }
        View::Contract => {
            // The contract on the left, what actually came back on the right: the two
            // halves of the same question, side by side so a declaration and a
            // measurement are never read as the same thing.
            let left = (width as f64 * 0.42) as usize;
            panels::contract(canvas, Rect::new(0, body_y, left, body_h), observation);
            panels::outcomes(
                canvas,
                Rect::new(left as i64, body_y, width - left, body_h),
                observation,
            );
        }
        View::Benchmark => {
            panels::benchmark(canvas, Rect::new(0, body_y, width, body_h), observation);
        }
        View::Economics => {
            let left = (width as f64 * 0.55) as usize;
            panels::economics(canvas, Rect::new(0, body_y, left, body_h), observation);
            panels::outcomes(
                canvas,
                Rect::new(left as i64, body_y, width - left, body_h),
                observation,
            );
        }
        View::Dashboard => {
            let left_w = (width as f64 * 0.44) as usize;
            let right_w = width - left_w;
            let top_h = (body_h as f64 * 0.58) as usize;
            let bottom_h = body_h - top_h;

            // The tornado owns the largest cell — it is the headline view.
            let tornado_rect = Rect::new(0, body_y, left_w, top_h);
            canvas.frame(
                tornado_rect.x,
                tornado_rect.y,
                tornado_rect.w,
                tornado_rect.h,
                "ACTIVE NODE TORNADO",
                theme::FRAME,
            );
            let tornado_inner = tornado_rect.inner();
            let clip = canvas.set_clip(
                tornado_inner.x,
                tornado_inner.y,
                tornado_inner.w,
                tornado_inner.h,
            );
            observation.tornado().draw_fitted(
                canvas,
                camera,
                Viewport::new(
                    tornado_rect.center_x(),
                    tornado_rect.center_y(),
                    tornado_inner.w,
                    tornado_inner.h,
                ),
                time,
            );
            canvas.restore_clip(clip);

            panels::equations(
                canvas,
                Rect::new(left_w as i64, body_y, right_w, top_h),
                observation,
            );

            let graph_w = right_w / 2;
            panels::node_graph(
                canvas,
                Rect::new(0, body_y + top_h as i64, left_w, bottom_h),
                observation,
                camera,
            );
            panels::roster(
                canvas,
                Rect::new(left_w as i64, body_y + top_h as i64, right_w - graph_w, bottom_h),
                observation,
            );
            panels::trust_matrix(
                canvas,
                Rect::new(
                    (left_w + right_w - graph_w) as i64,
                    body_y + top_h as i64,
                    graph_w,
                    bottom_h,
                ),
                observation,
            );
        }
    }

    let footer = format!(
        " view:{}  {}  intensity {:.2}  hubs {}  isolated {} ",
        view.as_str(),
        view.question(),
        observation.intensity(),
        observation.hubs.len(),
        observation.isolated.len()
    );
    canvas.text_clipped(1, height as i64 - 1, &footer, theme::LABEL, width.saturating_sub(2));
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::TaskSpec;
    use hatcher_neural::{pipeline, NeuralMesh};

    fn observation() -> Observation {
        let mut mesh = NeuralMesh::default();
        let task = TaskSpec::new("t", "layout", "rust")
            .with_features(vec![0.2, 0.9, 0.4, 0.6])
            .parsed_from_features();
        let trace = pipeline::run(&mut mesh, &task);
        Observation::capture(&mesh, &[0.2, 0.9, 0.4, 0.6], true, Some(trace))
    }

    #[test]
    fn every_view_renders_a_well_formed_frame() {
        let observation = observation();
        let camera = Camera::default();
        for view in View::ALL {
            let mut canvas = Canvas::new(140, 40);
            draw(&mut canvas, view, &observation, &camera, 1.5, 3);
            let plain = canvas.render_plain();
            assert_eq!(plain.lines().count(), 40, "{view:?}");
            assert!(plain.lines().all(|l| l.chars().count() == 140), "{view:?} overflowed");
            assert!(plain.chars().any(|c| !c.is_whitespace()), "{view:?} was blank");
            assert!(plain.contains(view.as_str()), "{view:?} footer missing");
        }
    }

    #[test]
    fn the_tab_strip_lights_the_active_view_and_lists_the_rest() {
        let observation = observation();
        let camera = Camera::default();
        let mut canvas = Canvas::new(140, 40);
        draw(&mut canvas, View::Contract, &observation, &camera, 0.0, 0);

        let strip = canvas.render_plain().lines().nth(1).unwrap().to_string();
        assert!(strip.contains("[contract]"), "the active tab is bracketed: {strip:?}");
        assert!(strip.contains(" benchmark "), "and the others are still listed");
        assert!(strip.contains(" dashboard "));
    }

    #[test]
    fn the_contract_view_says_when_nothing_real_has_been_reported() {
        let observation = observation();
        let camera = Camera::default();
        let mut canvas = Canvas::new(140, 40);
        draw(&mut canvas, View::Contract, &observation, &camera, 0.0, 0);
        let plain = canvas.render_plain();

        assert!(plain.contains("simulated"), "a rehearsing mesh must say so");
        assert!(plain.contains("0% of cohort"), "and admit nothing has been measured");
        assert!(plain.contains("THE FOUR VERBS"));
    }

    #[test]
    fn the_benchmark_view_says_so_when_no_benchmark_was_run() {
        let observation = observation();
        let camera = Camera::default();
        let mut canvas = Canvas::new(140, 40);
        draw(&mut canvas, View::Benchmark, &observation, &camera, 0.0, 0);
        assert!(canvas.render_plain().contains("--benchmark"));
    }

    #[test]
    fn the_economics_view_marks_declared_numbers_as_declared() {
        let observation = observation();
        let camera = Camera::default();
        let mut canvas = Canvas::new(140, 40);
        draw(&mut canvas, View::Economics, &observation, &camera, 0.0, 0);
        let plain = canvas.render_plain();

        assert!(plain.contains("declared"), "an unmeasured cohort must not look measured");
        assert!(plain.contains("COST & LATENCY"));
    }

    #[test]
    fn tiny_terminals_do_not_panic() {
        let observation = observation();
        let camera = Camera::default();
        for (w, h) in [(1usize, 1usize), (4, 2), (10, 3), (20, 6), (40, 10)] {
            let mut canvas = Canvas::new(w, h);
            draw(&mut canvas, View::Dashboard, &observation, &camera, 0.0, 0);
        }
    }

    #[test]
    fn view_names_round_trip() {
        for view in View::ALL {
            assert_eq!(View::parse(view.as_str()), Some(view));
            assert!(!view.question().is_empty());
        }
        assert_eq!(View::parse("VORTEX"), Some(View::Tornado));
        assert_eq!(View::parse("adapter"), Some(View::Contract));
        assert_eq!(View::parse("bench"), Some(View::Benchmark));
        assert_eq!(View::parse("cost"), Some(View::Economics));
        assert_eq!(View::parse("nope"), None);
    }

    #[test]
    fn panel_contents_never_bleed_across_a_border() {
        let observation = observation();
        let camera = Camera::default();
        let mut canvas = Canvas::new(120, 34);
        draw(&mut canvas, View::Dashboard, &observation, &camera, 1.0, 1);
        let plain = canvas.render_plain();

        // The tornado and node graph project geometry freely; if clipping
        // regressed, particles and labels land on the vertical rules between
        // panels. Assert the seam column stays structural.
        let left_w = (120.0f64 * 0.44) as usize;
        let seam = left_w - 1;
        // Rows 0 and 1 are the header and the tab strip; the last row is the footer.
        for line in plain.lines().skip(2).take(31) {
            let ch = line.chars().nth(seam).unwrap();
            assert!(
                matches!(ch, '│' | '─' | '╭' | '╮' | '╰' | '╯' | '┼' | ' '),
                "panel content leaked onto the seam column: {ch:?} in {line:?}"
            );
        }
    }

    #[test]
    fn the_frame_changes_as_the_vortex_turns() {
        let observation = observation();
        let camera = Camera::default();
        let mut a = Canvas::new(100, 30);
        let mut b = Canvas::new(100, 30);
        draw(&mut a, View::Tornado, &observation, &camera, 0.0, 0);
        draw(&mut b, View::Tornado, &observation, &camera, 2.0, 1);
        assert_ne!(a.render_plain(), b.render_plain());
    }
}
