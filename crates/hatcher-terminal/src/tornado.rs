//! The active-node tornado.
//!
//! When the mesh is working, the observatory spins the cohort up into a vortex:
//! each agent becomes an orbiting node whose radius, height, and angular speed
//! are read from its own state, and a cloud of tracer particles is carried
//! around the same field so the rotation reads as a funnel rather than a ring.
//!
//! The physics is deliberately literal rather than decorative:
//!
//! * **Radius** — inverse capability. Strong agents are drawn to the axis, weak
//!   ones fling out to the rim, so the eye lands on the core of the mesh first.
//! * **Height** — trust standing. Trusted hubs ride high in the funnel.
//! * **Angular speed** — activation. A node carrying live signal spins faster,
//!   which is what makes an active mesh visibly accelerate.
//! * **Amplitude** — `Ω`. A mesh that is getting smarter builds a taller, wider
//!   funnel; a collapsing one flattens toward a disc.
//!
//! An idle mesh must therefore look idle: with zero activation the funnel decays
//! to a slow, dim ring, and only spins up when work is actually flowing.

use crate::canvas::{Canvas, Viewport};
use crate::space::{Camera, Vec3};
use crate::theme::{self, Rgb};

/// Height of the funnel in world units at unit amplitude.
const FUNNEL_HEIGHT: f64 = 5.0;
/// Rim radius at the top of the funnel, at unit amplitude.
const FUNNEL_RADIUS: f64 = 3.4;
/// Radius at the very bottom of the funnel — the tornado's touchdown point.
const TOUCHDOWN_RADIUS: f64 = 0.35;
/// Tracer particles carried by the field.
const TRACERS: usize = 320;
/// Angular speed floor, so an idle mesh still drifts instead of freezing solid.
const IDLE_SPIN: f64 = 0.18;

/// One agent, resolved into vortex coordinates.
#[derive(Debug, Clone)]
pub struct VortexNode {
    pub id: String,
    pub label: String,
    /// Normalized capability `A_i`, drives radius and colour.
    pub capability: f64,
    /// Normalized trust standing, drives height in the funnel.
    pub trust: f64,
    /// Live activation from message passing, drives spin and brightness.
    pub activation: f64,
    /// Phase offset so the cohort is spread around the funnel.
    pub phase: f64,
}

/// The whole vortex: cohort plus the global terms that shape the funnel.
#[derive(Debug, Clone)]
pub struct Tornado {
    pub nodes: Vec<VortexNode>,
    /// `Ω`-derived amplitude in `[0, 1]`: how tall and wide the funnel stands.
    pub amplitude: f64,
    /// Mean activation across the cohort — the "is the mesh live" signal.
    pub intensity: f64,
    /// Whether the mesh is currently executing. A dormant mesh does not spin up.
    pub active: bool,
}

impl Tornado {
    /// Radius of the funnel wall at normalized height `t` in `[0, 1]`.
    ///
    /// The profile is quadratic, which is what gives the silhouette its
    /// characteristic concave flare rather than a straight cone.
    pub fn radius_at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        let amplitude = self.amplitude.clamp(0.0, 1.0);
        let profile = TOUCHDOWN_RADIUS + (FUNNEL_RADIUS - TOUCHDOWN_RADIUS) * t * t;
        profile * (0.45 + 0.55 * amplitude)
    }

    /// Height of the funnel in world units.
    pub fn height(&self) -> f64 {
        FUNNEL_HEIGHT * (0.5 + 0.5 * self.amplitude.clamp(0.0, 1.0))
    }

    /// Angular speed at normalized height `t`.
    ///
    /// Speed falls off with height: the touchdown point whips around fastest,
    /// which is what sells the shape as a vortex under rotation.
    pub fn spin_at(&self, t: f64) -> f64 {
        let base = if self.active { IDLE_SPIN + 2.4 * self.intensity.clamp(0.0, 1.0) } else { IDLE_SPIN };
        base * (1.9 - 0.9 * t.clamp(0.0, 1.0))
    }

    /// World position of a node at time `time`.
    pub fn node_position(&self, node: &VortexNode, time: f64) -> Vec3 {
        // Strong agents sit near the axis; weak ones are thrown to the rim.
        let t = (1.0 - node.capability.clamp(0.0, 1.0)) * 0.75 + (1.0 - node.trust.clamp(0.0, 1.0)) * 0.25;
        let radius = self.radius_at(t);
        let angle = node.phase + time * self.spin_at(t) * (0.6 + 0.8 * node.activation.clamp(0.0, 1.0));
        let y = -self.height() * 0.5 + self.height() * t;
        Vec3::new(radius * angle.cos(), y, radius * angle.sin())
    }

    /// Draw the vortex into `canvas`, centred on the given cell.
    ///
    /// `viewport` carries the panel's centre and inner dimensions in cells; the camera's
    /// focal length is derived from them so the funnel fills whatever space it
    /// is given instead of shrinking into the middle of a large panel.
    pub fn draw_fitted(&self, canvas: &mut Canvas, camera: &Camera, viewport: Viewport, time: f64) {
        let mut fitted = *camera;
        fitted.focal = self.fit_focal(camera, viewport.w, viewport.h);
        self.draw(canvas, &fitted, viewport.cx, viewport.cy, time);
    }

    /// Focal length that makes the funnel's bounding box fill the viewport.
    ///
    /// Solved against both axes and the tighter one wins, so the vortex is never
    /// clipped by the panel it is drawn into.
    fn fit_focal(&self, camera: &Camera, view_w: usize, view_h: usize) -> f64 {
        let radius = self.radius_at(1.0).max(0.05);
        let half_height = (self.height() * 0.5).max(0.05);
        let distance = camera.distance.max(0.1);

        // Leave a small margin so labels and the rim glyph stay inside.
        let usable_w = (view_w as f64 * 0.44).max(1.0);
        let usable_h = (view_h as f64 * 0.46).max(1.0);

        let focal_w = usable_w * distance / (radius * crate::space::CELL_ASPECT);
        let focal_h = usable_h * distance / half_height;
        focal_w.min(focal_h).clamp(4.0, 80.0)
    }

    /// Draw the vortex into `canvas`, centred on the given cell.
    pub fn draw(&self, canvas: &mut Canvas, camera: &Camera, cx: f64, cy: f64, time: f64) {
        self.draw_tracers(canvas, camera, cx, cy, time);
        self.draw_axis(canvas, camera, cx, cy);
        self.draw_nodes(canvas, camera, cx, cy, time);
    }

    /// The carried particle field that makes the funnel legible as a surface.
    ///
    /// Particles are laid out as a stack of rings rather than one sequence: a
    /// single low-discrepancy walk over both height and angle produces a thin
    /// helix, which reads as a spring, not a vortex. Rings give a surface.
    fn draw_tracers(&self, canvas: &mut Canvas, camera: &Camera, cx: f64, cy: f64, time: f64) {
        let budget = if self.active { TRACERS } else { TRACERS / 3 };
        let rings = if self.active { 22 } else { 10 };
        let per_ring = (budget / rings).max(4);

        for ring in 0..rings {
            // Bias sampling toward the top so the flared rim keeps its density
            // as it widens, instead of thinning out where there is most area.
            let t = ((ring as f64 + 0.5) / rings as f64).powf(0.85);
            let y = -self.height() * 0.5 + self.height() * t;
            let base_radius = self.radius_at(t);
            let spin = self.spin_at(t);

            // Offset each ring so the funnel wall does not look like stacked hoops.
            let ring_offset = ring as f64 * 2.399_963_23;

            for slot in 0..per_ring {
                let seed = (ring * per_ring + slot) as f64;
                let angle = ring_offset + slot as f64 * std::f64::consts::TAU / per_ring as f64 + time * spin;
                // Break the ring into turbulence so the wall has thickness.
                let jitter = 0.86 + 0.14 * ((seed * 0.754_877_666) % 1.0);
                let radius = base_radius * jitter;

                let point = Vec3::new(radius * angle.cos(), y, radius * angle.sin());
                let Some(p) = camera.project(point, cx, cy) else {
                    continue;
                };

                // Depth cue: far side of the funnel dims, so rotation reads as 3D.
                let depth_cue = (p.scale * 1.6).clamp(0.18, 1.0);
                let energy = (self.intensity * 0.6 + t * 0.4).clamp(0.0, 1.0);
                let colour = theme::heat(energy).dim(depth_cue);
                // Spread the glyph ramp across the full depth range, so the near
                // wall reads solid and the far wall reads faint. Compressing this
                // into a narrow band makes the whole funnel one flat texture.
                let ch = if self.active {
                    theme::glyph(0.15 + 0.85 * (0.35 * energy + 0.65 * depth_cue))
                } else {
                    '·'
                };
                canvas.put_depth(p.x, p.y, ch, colour, p.depth);
            }
        }
    }

    /// The rotation axis, drawn faintly so the funnel has a spine to read against.
    fn draw_axis(&self, canvas: &mut Canvas, camera: &Camera, cx: f64, cy: f64) {
        let half = self.height() * 0.5;
        let top = camera.project(Vec3::new(0.0, half, 0.0), cx, cy);
        let bottom = camera.project(Vec3::new(0.0, -half, 0.0), cx, cy);
        if let (Some(a), Some(b)) = (top, bottom) {
            canvas.line(a.x, a.y, b.x, b.y, '│', theme::FRAME, (a.depth + b.depth) * 0.5);
        }
    }

    /// The agents themselves, drawn over the field with their labels.
    fn draw_nodes(&self, canvas: &mut Canvas, camera: &Camera, cx: f64, cy: f64, time: f64) {
        for node in &self.nodes {
            let position = self.node_position(node, time);
            let Some(p) = camera.project(position, cx, cy) else {
                continue;
            };

            let activation = node.activation.clamp(0.0, 1.0);
            let colour: Rgb = theme::heat(activation).dim((p.scale * 1.8).clamp(0.35, 1.0));
            // Hot nodes get a solid marker; quiet ones stay small and unobtrusive.
            let marker = if activation > 0.66 {
                '◉'
            } else if activation > 0.33 {
                '◍'
            } else {
                '○'
            };
            canvas.put_depth(p.x, p.y, marker, colour, p.depth - 0.01);

            // Label only the nodes carrying real signal, or the panel turns to soup.
            if activation > 0.4 {
                let label: String = node.label.chars().take(9).collect();
                canvas.text_clipped(p.x + 2, p.y, &label, theme::LABEL, 10);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tornado(active: bool, intensity: f64) -> Tornado {
        Tornado {
            nodes: vec![
                VortexNode {
                    id: "a".into(),
                    label: "alpha".into(),
                    capability: 0.9,
                    trust: 0.8,
                    activation: 0.9,
                    phase: 0.0,
                },
                VortexNode {
                    id: "b".into(),
                    label: "beta".into(),
                    capability: 0.1,
                    trust: 0.2,
                    activation: 0.1,
                    phase: 1.0,
                },
            ],
            amplitude: 0.7,
            intensity,
            active,
        }
    }

    #[test]
    fn the_funnel_narrows_toward_touchdown() {
        let t = tornado(true, 0.8);
        assert!(t.radius_at(0.0) < t.radius_at(0.5));
        assert!(t.radius_at(0.5) < t.radius_at(1.0));
    }

    #[test]
    fn an_active_mesh_spins_faster_than_an_idle_one() {
        let active = tornado(true, 0.9);
        let idle = tornado(false, 0.0);
        assert!(active.spin_at(0.5) > idle.spin_at(0.5));
        assert!(idle.spin_at(0.5) > 0.0, "an idle mesh drifts, it does not freeze");
    }

    #[test]
    fn touchdown_whips_faster_than_the_rim() {
        let t = tornado(true, 0.6);
        assert!(t.spin_at(0.0) > t.spin_at(1.0));
    }

    #[test]
    fn capable_agents_orbit_nearer_the_axis() {
        let t = tornado(true, 0.5);
        let strong = t.node_position(&t.nodes[0], 0.0);
        let weak = t.node_position(&t.nodes[1], 0.0);
        let radius = |v: Vec3| (v.x * v.x + v.z * v.z).sqrt();
        assert!(radius(strong) < radius(weak));
        assert!(strong.y < weak.y, "a strong, trusted agent sits low in the funnel core");
    }

    #[test]
    fn amplitude_scales_the_funnel() {
        let mut small = tornado(true, 0.5);
        small.amplitude = 0.0;
        let mut large = tornado(true, 0.5);
        large.amplitude = 1.0;
        assert!(large.height() > small.height());
        assert!(large.radius_at(1.0) > small.radius_at(1.0));
    }

    #[test]
    fn nodes_actually_rotate_over_time() {
        let t = tornado(true, 0.9);
        let start = t.node_position(&t.nodes[0], 0.0);
        let later = t.node_position(&t.nodes[0], 1.0);
        assert!(start != later, "an active vortex must move between frames");
    }

    #[test]
    fn a_fitted_vortex_actually_fills_its_viewport() {
        let t = tornado(true, 0.9);
        let camera = Camera::default();

        // Count how much of the panel the funnel covers; a vortex that shrinks
        // into the middle of a large panel is the regression this guards.
        let mut canvas = Canvas::new(60, 24);
        t.draw_fitted(&mut canvas, &camera, Viewport::new(30.0, 12.0, 60, 24), 1.0);
        let plain = canvas.render_plain();

        let (mut min_x, mut max_x, mut min_y, mut max_y) = (usize::MAX, 0usize, usize::MAX, 0usize);
        for (y, line) in plain.lines().enumerate() {
            for (x, ch) in line.chars().enumerate() {
                if !ch.is_whitespace() {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        assert!(min_x != usize::MAX, "the fitted vortex drew nothing");
        let span_x = max_x - min_x;
        let span_y = max_y - min_y;
        assert!(span_x >= 30, "vortex only spans {span_x} of 60 columns");
        assert!(span_y >= 12, "vortex only spans {span_y} of 24 rows");
    }

    #[test]
    fn fitting_scales_with_the_viewport() {
        let t = tornado(true, 0.8);
        let camera = Camera::default();
        let small = t.fit_focal(&camera, 40, 12);
        let large = t.fit_focal(&camera, 120, 40);
        assert!(large > small, "a bigger panel must earn a longer focal length");
    }

    #[test]
    fn drawing_marks_the_canvas_and_stays_inside_it() {
        let mut canvas = Canvas::new(60, 24);
        let camera = Camera::default();
        tornado(true, 0.9).draw(&mut canvas, &camera, 30.0, 12.0, 1.0);
        let plain = canvas.render_plain();
        assert!(plain.chars().any(|c| !c.is_whitespace()), "the vortex must render something");
        assert_eq!(plain.lines().count(), 24);
        assert!(plain.lines().all(|line| line.chars().count() == 60));
    }
}
