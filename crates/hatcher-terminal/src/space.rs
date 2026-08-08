//! Minimal 3D: vectors, a yaw/pitch orbit camera, and perspective projection
//! into character-cell space.
//!
//! Terminal cells are roughly twice as tall as they are wide, so the projection
//! applies a fixed aspect correction — without it every sphere in the
//! observatory renders as an egg.

/// Character cells are about twice as tall as wide; scale x to compensate.
pub const CELL_ASPECT: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);

    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn scale(self, k: f64) -> Vec3 {
        Vec3::new(self.x * k, self.y * k, self.z * k)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, k: f64) -> Vec3 {
        self.scale(k)
    }
}

/// A projected point in canvas space.
#[derive(Debug, Clone, Copy)]
pub struct Projected {
    pub x: i64,
    pub y: i64,
    /// Camera-space depth, for the z-buffer. Smaller is nearer.
    pub depth: f64,
    /// `1/depth`-style scale in `(0, 1]`, for size and brightness cueing.
    pub scale: f64,
}

/// An orbit camera looking at the origin.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Rotation about the vertical axis, radians.
    pub yaw: f64,
    /// Tilt above the horizon, radians.
    pub pitch: f64,
    /// Distance from the origin, world units. Clamped away from zero.
    pub distance: f64,
    /// Focal length in cell units.
    pub focal: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.32,
            distance: 9.0,
            focal: 14.0,
        }
    }
}

impl Camera {
    /// Rotate the world into camera space: yaw about Y, then pitch about X, then
    /// push back along Z by `distance`.
    pub fn to_camera_space(&self, point: Vec3) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let x = point.x * cy + point.z * sy;
        let z = -point.x * sy + point.z * cy;

        let (sp, cp) = self.pitch.sin_cos();
        let y = point.y * cp - z * sp;
        let z = point.y * sp + z * cp;

        Vec3::new(x, y, z + self.distance.max(0.1))
    }

    /// Project a world point into canvas cells centred on `(cx, cy)`.
    ///
    /// Returns `None` for points at or behind the near plane, which the caller
    /// should simply skip.
    pub fn project(&self, point: Vec3, cx: f64, cy: f64) -> Option<Projected> {
        let camera_space = self.to_camera_space(point);
        const NEAR: f64 = 0.35;
        if camera_space.z <= NEAR {
            return None;
        }

        let inverse = self.focal / camera_space.z;
        let sx = cx + camera_space.x * inverse * CELL_ASPECT;
        // Canvas y grows downward; world y grows upward.
        let sy = cy - camera_space.y * inverse;
        if !sx.is_finite() || !sy.is_finite() {
            return None;
        }

        Some(Projected {
            x: sx.round() as i64,
            y: sy.round() as i64,
            depth: camera_space.z,
            scale: (inverse / self.focal).clamp(0.0, 1.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_origin_projects_to_the_canvas_centre() {
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            ..Camera::default()
        };
        let p = camera.project(Vec3::ZERO, 40.0, 12.0).expect("origin is visible");
        assert_eq!((p.x, p.y), (40, 12));
        assert!((p.depth - camera.distance).abs() < 1e-9);
    }

    #[test]
    fn nearer_points_have_smaller_depth() {
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            ..Camera::default()
        };
        let near = camera.project(Vec3::new(0.0, 0.0, -2.0), 40.0, 12.0).unwrap();
        let far = camera.project(Vec3::new(0.0, 0.0, 2.0), 40.0, 12.0).unwrap();
        assert!(near.depth < far.depth);
        assert!(near.scale > far.scale);
    }

    #[test]
    fn points_behind_the_near_plane_are_culled() {
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            distance: 1.0,
            focal: 14.0,
        };
        assert!(camera.project(Vec3::new(0.0, 0.0, -5.0), 40.0, 12.0).is_none());
    }

    #[test]
    fn up_in_the_world_is_up_on_screen() {
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            ..Camera::default()
        };
        let high = camera.project(Vec3::new(0.0, 1.0, 0.0), 40.0, 12.0).unwrap();
        assert!(high.y < 12, "positive world y must render above centre");
    }

    #[test]
    fn a_half_turn_of_yaw_mirrors_x() {
        let front = Camera {
            yaw: 0.0,
            pitch: 0.0,
            ..Camera::default()
        };
        let back = Camera {
            yaw: std::f64::consts::PI,
            pitch: 0.0,
            ..Camera::default()
        };
        let a = front.project(Vec3::new(1.0, 0.0, 0.0), 40.0, 12.0).unwrap();
        let b = back.project(Vec3::new(1.0, 0.0, 0.0), 40.0, 12.0).unwrap();
        assert_eq!(a.x - 40, -(b.x - 40));
    }

    #[test]
    fn vector_maths_is_sane() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 1e-12);
        assert_eq!(v - v, Vec3::ZERO);
        assert_eq!(v * 2.0, Vec3::new(6.0, 8.0, 0.0));
        assert_eq!(v.scale(2.0), Vec3::new(6.0, 8.0, 0.0));
        assert_eq!(v + Vec3::new(1.0, 1.0, 1.0), Vec3::new(4.0, 5.0, 1.0));
    }
}
