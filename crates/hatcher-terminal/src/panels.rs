//! The panels.
//!
//! Each function draws one view of the mesh into a rectangle of the canvas.
//! They are deliberately independent: the dashboard is just a layout over these,
//! and each can be exercised on its own in a test or a `--view` invocation.

use hatcher_core::PipelineStage;

use crate::canvas::Canvas;
use crate::observe::Observation;
use crate::space::{Camera, Vec3};
use crate::theme::{self, Rgb};

/// A rectangle of canvas, in cells.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub const fn new(x: i64, y: i64, w: usize, h: usize) -> Self {
        Self { x, y, w, h }
    }

    /// The drawable area inside the frame border.
    pub fn inner(&self) -> Rect {
        Rect::new(self.x + 2, self.y + 1, self.w.saturating_sub(4), self.h.saturating_sub(2))
    }

    pub fn center_x(&self) -> f64 {
        self.x as f64 + self.w as f64 / 2.0
    }

    pub fn center_y(&self) -> f64 {
        self.y as f64 + self.h as f64 / 2.0
    }
}

/// The 3D node graph: agents placed on a sphere, edges drawn by trust weight.
///
/// Placement uses a Fibonacci sphere so the cohort is spread evenly regardless
/// of size, and every node keeps a stable position between frames — the graph
/// rotates, it does not reshuffle.
pub fn node_graph(canvas: &mut Canvas, rect: Rect, observation: &Observation, camera: &Camera) {
    canvas.frame(rect.x, rect.y, rect.w, rect.h, "MESH NODE GRAPH", theme::FRAME);
    let inner = rect.inner();
    if inner.w < 8 || inner.h < 4 || observation.agents.is_empty() {
        return;
    }
    // Everything below draws projected geometry, which has no idea where the
    // panel border is; clip so it cannot bleed into the neighbouring panel.
    let clip = canvas.set_clip(inner.x, inner.y, inner.w, inner.h);

    let cx = rect.center_x();
    let cy = rect.center_y();
    let count = observation.agents.len();
    let positions: Vec<Vec3> = (0..count).map(|index| fibonacci_sphere(index, count, 3.1)).collect();

    // Edges first, so nodes always sit on top of their own connections.
    for edge in &observation.edges {
        let (Some(from), Some(to)) = (index_of(observation, &edge.from), index_of(observation, &edge.to)) else {
            continue;
        };
        // Only the meaningful half of a dense graph, or the panel is a solid block.
        if edge.weight < 0.35 || from >= to {
            continue;
        }
        let (Some(a), Some(b)) = (
            camera.project(positions[from], cx, cy),
            camera.project(positions[to], cx, cy),
        ) else {
            continue;
        };

        let depth = (a.depth + b.depth) * 0.5;
        let colour = theme::trust_ramp(edge.trust).dim((0.35 + edge.weight * 0.65).clamp(0.0, 1.0));
        let ch = if edge.weight > 0.75 { '═' } else { '·' };
        canvas.line(a.x, a.y, b.x, b.y, ch, colour, depth);
    }

    for (index, agent) in observation.agents.iter().enumerate() {
        let Some(p) = camera.project(positions[index], cx, cy) else {
            continue;
        };
        let colour = theme::heat(agent.capability_norm).dim((p.scale * 1.8).clamp(0.4, 1.0));
        let marker = if agent.activation > 0.6 { '◉' } else { '◎' };
        canvas.put_depth(p.x, p.y, marker, colour, p.depth - 0.01);
        canvas.text_clipped(p.x + 2, p.y, &agent.role, theme::LABEL, 9);
    }

    canvas.restore_clip(clip);
}

fn index_of(observation: &Observation, id: &str) -> Option<usize> {
    observation.agents.iter().position(|agent| agent.id == id)
}

/// Evenly distributed points on a sphere, deterministic in `index`.
fn fibonacci_sphere(index: usize, count: usize, radius: f64) -> Vec3 {
    let count = count.max(1) as f64;
    let i = index as f64;
    let y = 1.0 - 2.0 * (i + 0.5) / count;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = i * 2.399_963_23; // golden angle
    Vec3::new(radius * r * theta.cos(), radius * y, radius * r * theta.sin())
}

/// The trust matrix `T_ij` as a heat grid.
pub fn trust_matrix(canvas: &mut Canvas, rect: Rect, observation: &Observation) {
    canvas.frame(rect.x, rect.y, rect.w, rect.h, "TRUST MATRIX T(i,j)", theme::FRAME);
    let inner = rect.inner();
    if inner.w < 6 || inner.h < 3 {
        return;
    }

    let n = observation.agents.len();
    if n == 0 {
        return;
    }

    // Each cell is two columns wide so the grid reads square in a terminal.
    let cell_w = 2usize;
    let max_cols = (inner.w.saturating_sub(4)) / cell_w;
    let shown = n.min(max_cols).min(inner.h.saturating_sub(1));

    for row in 0..shown {
        let y = inner.y + row as i64;
        let label: String = observation.agents[row].id.chars().take(3).collect();
        canvas.text_clipped(inner.x, y, &label, theme::LABEL, 3);

        for column in 0..shown {
            let value = lookup_trust(observation, row, column);
            let x = inner.x + 4 + (column * cell_w) as i64;
            let colour = if row == column {
                theme::FRAME
            } else {
                theme::trust_ramp(value)
            };
            let ch = if row == column { '·' } else { theme::glyph(value) };
            canvas.put(x, y, ch, colour);
            canvas.put(x + 1, y, ch, colour);
        }
    }
}

fn lookup_trust(observation: &Observation, from: usize, to: usize) -> f64 {
    let (Some(a), Some(b)) = (observation.agents.get(from), observation.agents.get(to)) else {
        return 0.0;
    };
    observation
        .edges
        .iter()
        .find(|edge| edge.from == a.id && edge.to == b.id)
        .map(|edge| edge.trust)
        .unwrap_or(0.0)
}

/// The ten equations with their live values, as labelled meters.
pub fn equations(canvas: &mut Canvas, rect: Rect, observation: &Observation) {
    canvas.frame(rect.x, rect.y, rect.w, rect.h, "HAMNS EQUATIONS", theme::FRAME);
    let inner = rect.inner();
    if inner.w < 24 || inner.h < 3 {
        return;
    }

    let delta = observation
        .last_trace
        .as_ref()
        .map(|trace| trace.delta)
        .unwrap_or_default();

    let mean_capability = if observation.agents.is_empty() {
        0.0
    } else {
        observation.agents.iter().map(|a| a.capability_norm).sum::<f64>() / observation.agents.len() as f64
    };
    let priority = observation
        .last_trace
        .as_ref()
        .map(|trace| trace.priority.value)
        .unwrap_or(0.0);

    let rows: [(&str, f64, &str); 10] = [
        ("1 Ω  global", observation.omega_norm(), "Ω+α(L+E+C)−β(F+D)"),
        ("2 A_i capability", mean_capability, "I·S·P·C·M"),
        ("3 A  mesh", observation.emergence_ratio, "ΣA+γΣAAW"),
        ("4 T  trust", observation.mean_trust, "T+λS−μE"),
        ("5 P  priority", (priority / (priority + 1.0)).clamp(0.0, 1.0), "(U·B·I)/(C+τ)"),
        ("6 M  memory", delta.learning.clamp(0.0, 1.0), "M+ηK−δR"),
        ("7 C  confidence", observation.mean_confidence, "C+σ·ok−ρ·err"),
        ("8 S  special.", mean_capability, "S+κ·exp−ω·obs"),
        ("9 W  plasticity", observation.intensity(), "W+φT−ψ·lat"),
        ("10 R resource", 1.0 - delta.failure.clamp(0.0, 1.0), "P/(E+L)"),
    ];

    let label_w = 18usize;
    let bar_w = inner.w.saturating_sub(label_w + 8).clamp(4, 24);

    for (index, (label, value, formula)) in rows.iter().enumerate() {
        if index >= inner.h {
            break;
        }
        let y = inner.y + index as i64;
        canvas.text_clipped(inner.x, y, label, theme::TEXT, label_w);
        canvas.bar(inner.x + label_w as i64, y, bar_w, *value, theme::heat);
        let readout = format!(" {:.2}", value);
        canvas.text_clipped(inner.x + (label_w + bar_w) as i64, y, &readout, theme::LABEL, 6);

        let formula_x = inner.x + (label_w + bar_w + 6) as i64;
        let remaining = inner.w.saturating_sub(label_w + bar_w + 6);
        if remaining > 4 {
            canvas.text_clipped(formula_x, y, formula, theme::FRAME, remaining);
        }
    }
}

/// The ten-station pipeline as a flow strip, showing where the last trace went.
pub fn pipeline(canvas: &mut Canvas, rect: Rect, observation: &Observation) {
    canvas.frame(rect.x, rect.y, rect.w, rect.h, "PIPELINE TRACE", theme::FRAME);
    let inner = rect.inner();
    if inner.w < 12 || inner.h < 2 {
        return;
    }

    let Some(trace) = observation.last_trace.as_ref() else {
        canvas.text_clipped(inner.x, inner.y, "idle — no trace yet", theme::LABEL, inner.w);
        return;
    };

    for (index, stage) in PipelineStage::ALL.iter().enumerate() {
        if index >= inner.h {
            break;
        }
        let y = inner.y + index as i64;
        let record = trace.stages.get(index);
        let (marker, colour): (char, Rgb) = match record {
            Some(r) if r.success => ('✔', theme::GOOD),
            Some(_) => ('✘', theme::BAD),
            None => ('·', theme::FRAME),
        };

        canvas.put(inner.x, y, marker, colour);
        let name = format!("{stage:?}");
        canvas.text_clipped(inner.x + 2, y, &name, theme::TEXT, 13);

        if let Some(r) = record {
            let bar_x = inner.x + 16;
            let bar_w = inner.w.saturating_sub(24).clamp(3, 14);
            canvas.bar(bar_x, y, bar_w, r.confidence, theme::heat);
            let agent = r.agent.clone().unwrap_or_else(|| "—".into());
            canvas.text_clipped(bar_x + bar_w as i64 + 1, y, &agent, theme::LABEL, 8);
        }
    }
}

/// The cohort roster with capability, trust, and outcome counts.
pub fn roster(canvas: &mut Canvas, rect: Rect, observation: &Observation) {
    canvas.frame(rect.x, rect.y, rect.w, rect.h, "AGENT COHORT", theme::FRAME);
    let inner = rect.inner();
    if inner.w < 20 || inner.h < 2 {
        return;
    }

    canvas.text_clipped(inner.x, inner.y, "AGENT        CAP   TRUST  ACT   OK/TRY", theme::FRAME, inner.w);

    for (index, agent) in observation.ranked().iter().enumerate() {
        let y = inner.y + 1 + index as i64;
        if index + 1 >= inner.h {
            break;
        }
        let rate = agent
            .success_rate()
            .map(|r| format!("{:.0}%", r * 100.0))
            .unwrap_or_else(|| "—".into());
        let line = format!(
            "{:<12} {:.2}  {:.2}   {:.2}  {:>4} {}/{}",
            agent.label.chars().take(12).collect::<String>(),
            agent.capability_norm,
            agent.trust,
            agent.activation,
            rate,
            agent.successes,
            agent.attempts
        );
        let colour = theme::heat(agent.capability_norm);
        canvas.text_clipped(inner.x, y, &line, colour, inner.w);
    }
}

/// The header strip: the numbers you want without reading a panel.
pub fn header(canvas: &mut Canvas, width: usize, observation: &Observation, frame: u64) {
    let status = if observation.active { "● MESH ACTIVE" } else { "○ MESH IDLE" };
    let status_colour = if observation.active { theme::GOOD } else { theme::LABEL };
    canvas.text(1, 0, status, status_colour);

    let line = format!(
        "Ω {:.4}  epoch {}  A {:.2} (raw {:.2} + emergent {:.2})  trust {:.2}  E {:.0}%  frame {}",
        observation.omega,
        observation.epoch,
        observation.intelligence.total,
        observation.intelligence.raw,
        observation.intelligence.emergent,
        observation.mean_trust,
        observation.emergence_ratio * 100.0,
        frame
    );
    canvas.text_clipped(17, 0, &line, theme::TEXT, width.saturating_sub(18));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Observation;
    use hatcher_core::TaskSpec;
    use hatcher_neural::{pipeline as neural_pipeline, NeuralMesh};

    fn observation(active: bool) -> Observation {
        let mut mesh = NeuralMesh::default();
        let task = TaskSpec::new("t", "panel test", "rust")
            .with_features(vec![0.4, 0.6, 0.3, 0.8])
            .parsed_from_features();
        let trace = neural_pipeline::run(&mut mesh, &task);
        Observation::capture(&mesh, &[0.4, 0.6, 0.3, 0.8], active, Some(trace))
    }

    fn drew_something(canvas: &Canvas) -> bool {
        canvas.render_plain().chars().any(|c| !c.is_whitespace())
    }

    #[test]
    fn every_panel_draws_within_its_rectangle() {
        let observation = observation(true);
        let camera = Camera::default();
        type PanelFn = fn(&mut Canvas, Rect, &Observation);
        let panels: Vec<(&str, PanelFn)> = vec![
            ("trust", trust_matrix),
            ("equations", equations),
            ("pipeline", pipeline),
            ("roster", roster),
        ];

        for (name, draw) in panels {
            let mut canvas = Canvas::new(80, 24);
            draw(&mut canvas, Rect::new(0, 0, 80, 24), &observation);
            assert!(drew_something(&canvas), "{name} drew nothing");
            let plain = canvas.render_plain();
            assert_eq!(plain.lines().count(), 24, "{name} changed the canvas height");
            assert!(plain.lines().all(|l| l.chars().count() == 80), "{name} overflowed");
        }

        let mut canvas = Canvas::new(80, 24);
        node_graph(&mut canvas, Rect::new(0, 0, 80, 24), &observation, &camera);
        assert!(drew_something(&canvas));
    }

    #[test]
    fn panels_survive_being_squeezed_to_nothing() {
        let observation = observation(false);
        let camera = Camera::default();
        for size in [(0usize, 0usize), (1, 1), (3, 2), (6, 3)] {
            let mut canvas = Canvas::new(size.0.max(1), size.1.max(1));
            let rect = Rect::new(0, 0, size.0, size.1);
            trust_matrix(&mut canvas, rect, &observation);
            equations(&mut canvas, rect, &observation);
            pipeline(&mut canvas, rect, &observation);
            roster(&mut canvas, rect, &observation);
            node_graph(&mut canvas, rect, &observation, &camera);
        }
    }

    #[test]
    fn the_pipeline_panel_says_so_when_there_is_no_trace() {
        let mesh = NeuralMesh::default();
        let idle = Observation::capture(&mesh, &[0.5], false, None);
        let mut canvas = Canvas::new(40, 6);
        pipeline(&mut canvas, Rect::new(0, 0, 40, 6), &idle);
        assert!(canvas.render_plain().contains("idle"));
    }

    #[test]
    fn the_header_reports_active_and_idle_differently() {
        let mut active_canvas = Canvas::new(100, 1);
        header(&mut active_canvas, 100, &observation(true), 7);
        assert!(active_canvas.render_plain().contains("ACTIVE"));

        let mut idle_canvas = Canvas::new(100, 1);
        header(&mut idle_canvas, 100, &observation(false), 7);
        assert!(idle_canvas.render_plain().contains("IDLE"));
    }

    #[test]
    fn fibonacci_placement_is_stable_and_spread_out() {
        let a = fibonacci_sphere(3, 9, 3.0);
        let b = fibonacci_sphere(3, 9, 3.0);
        assert_eq!(a, b, "placement must not move between frames");
        assert!((a.length() - 3.0).abs() < 1e-9, "points must lie on the sphere");
        assert!(fibonacci_sphere(0, 9, 3.0) != fibonacci_sphere(1, 9, 3.0));
    }

    #[test]
    fn rect_inner_never_underflows() {
        let inner = Rect::new(0, 0, 1, 1).inner();
        assert_eq!((inner.w, inner.h), (0, 0));
    }
}
