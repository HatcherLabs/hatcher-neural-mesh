//! Layout: composing the panels into one frame.

use crate::canvas::{Canvas, Viewport};
use crate::observe::Observation;
use crate::panels::{self, Rect};
use crate::space::Camera;
use crate::theme;

/// Which view the observatory is showing.
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
}

impl View {
    pub fn parse(value: &str) -> Option<View> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dashboard" | "all" => Some(View::Dashboard),
            "tornado" | "vortex" => Some(View::Tornado),
            "graph" | "nodes" => Some(View::Graph),
            "equations" | "eq" => Some(View::Equations),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            View::Dashboard => "dashboard",
            View::Tornado => "tornado",
            View::Graph => "graph",
            View::Equations => "equations",
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

    let body_y = 1i64;
    let body_h = height.saturating_sub(2);
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
        " view:{}  intensity {:.2}  hubs {}  isolated {} ",
        view.as_str(),
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
        for view in [View::Dashboard, View::Tornado, View::Graph, View::Equations] {
            let mut canvas = Canvas::new(120, 34);
            draw(&mut canvas, view, &observation, &camera, 1.5, 3);
            let plain = canvas.render_plain();
            assert_eq!(plain.lines().count(), 34, "{view:?}");
            assert!(plain.lines().all(|l| l.chars().count() == 120), "{view:?} overflowed");
            assert!(plain.chars().any(|c| !c.is_whitespace()), "{view:?} was blank");
            assert!(plain.contains(view.as_str()), "{view:?} footer missing");
        }
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
        for view in [View::Dashboard, View::Tornado, View::Graph, View::Equations] {
            assert_eq!(View::parse(view.as_str()), Some(view));
        }
        assert_eq!(View::parse("VORTEX"), Some(View::Tornado));
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
        for line in plain.lines().skip(1).take(32) {
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
