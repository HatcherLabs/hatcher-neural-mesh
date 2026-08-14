//! Colour vocabulary for the observatory.
//!
//! Every visual channel in this crate resolves to an [`Rgb`] through one of the
//! ramps here, so "hot" means the same thing on the tornado as it does on the
//! trust matrix and the reader can compare two panels by colour alone.

/// A 24-bit terminal colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    /// Linear blend, `t` clamped to `[0, 1]`.
    pub fn mix(self, other: Rgb, t: f64) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * t).round() as u8;
        Rgb(
            lerp(self.0, other.0),
            lerp(self.1, other.1),
            lerp(self.2, other.2),
        )
    }

    /// Scale toward black. Used for depth cueing in the 3D views.
    pub fn dim(self, factor: f64) -> Rgb {
        let f = factor.clamp(0.0, 1.0);
        Rgb(
            (self.0 as f64 * f).round() as u8,
            (self.1 as f64 * f).round() as u8,
            (self.2 as f64 * f).round() as u8,
        )
    }

    /// Perceptual weight in `[0, 1]`, used to decide when a pixel may be overdrawn.
    pub fn luma(self) -> f64 {
        (0.2126 * self.0 as f64 + 0.7152 * self.1 as f64 + 0.0722 * self.2 as f64) / 255.0
    }
}

pub const BG: Rgb = Rgb(6, 8, 14);
pub const FRAME: Rgb = Rgb(38, 52, 74);
pub const LABEL: Rgb = Rgb(126, 148, 178);
pub const TEXT: Rgb = Rgb(198, 214, 232);
pub const ACCENT: Rgb = Rgb(94, 234, 212);
pub const WARN: Rgb = Rgb(251, 191, 36);
pub const BAD: Rgb = Rgb(248, 113, 113);
pub const GOOD: Rgb = Rgb(74, 222, 128);

/// Cold → hot ramp. The crate's primary scalar encoding: use it for any value
/// already normalized to `[0, 1]`.
pub fn heat(t: f64) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    const STOPS: [(f64, Rgb); 5] = [
        (0.00, Rgb(18, 26, 58)),
        (0.30, Rgb(37, 99, 235)),
        (0.55, Rgb(45, 212, 191)),
        (0.78, Rgb(250, 204, 21)),
        (1.00, Rgb(244, 63, 94)),
    ];
    for pair in STOPS.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t <= t1 {
            let span = t1 - t0;
            let local = if span <= f64::EPSILON {
                0.0
            } else {
                (t - t0) / span
            };
            return c0.mix(c1, local);
        }
    }
    STOPS[STOPS.len() - 1].1
}

/// Ramp for trust values, where the midpoint is the neutral baseline rather
/// than "half hot": distrust reads red, earned trust reads green.
pub fn trust_ramp(t: f64) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        Rgb(120, 30, 46).mix(Rgb(70, 84, 105), t / 0.5)
    } else {
        Rgb(70, 84, 105).mix(GOOD, (t - 0.5) / 0.5)
    }
}

/// Glyph ramp by density, coarse → solid. Shared by every field renderer so a
/// denser glyph always means more of whatever the panel is measuring.
pub const DENSITY: [char; 10] = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

/// Pick a density glyph for `t` in `[0, 1]`.
pub fn glyph(t: f64) -> char {
    let t = t.clamp(0.0, 1.0);
    let index = (t * (DENSITY.len() - 1) as f64).round() as usize;
    DENSITY[index.min(DENSITY.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_is_monotone_in_luma_at_the_ends() {
        assert!(heat(0.0).luma() < heat(1.0).luma());
    }

    #[test]
    fn ramps_clamp_out_of_range_input() {
        assert_eq!(heat(-5.0), heat(0.0));
        assert_eq!(heat(5.0), heat(1.0));
        assert_eq!(trust_ramp(-1.0), trust_ramp(0.0));
        assert_eq!(glyph(9.0), '@');
        assert_eq!(glyph(-9.0), ' ');
    }

    #[test]
    fn trust_midpoint_is_neutral_not_hot() {
        assert_eq!(trust_ramp(0.5), Rgb(70, 84, 105));
    }

    #[test]
    fn mix_and_dim_stay_in_gamut() {
        let c = Rgb(255, 255, 255).mix(Rgb(0, 0, 0), 0.5);
        assert_eq!(c, Rgb(128, 128, 128));
        assert_eq!(Rgb(200, 100, 50).dim(0.0), Rgb(0, 0, 0));
    }
}
