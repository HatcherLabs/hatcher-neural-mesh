//! The Greek-letter coefficient set that tunes every HAMNS equation.
//!
//! This is the whole knob surface of the mesh, in one serializable struct, so the
//! Hatcher frontend's Config tab can read and write it as a single JSON document.

use serde::{Deserialize, Serialize};

/// Tuning coefficients for the ten HAMNS update rules.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MeshCoefficients {
    /// `α` — how strongly learning, emergence, and collaboration raise `Ω`.
    pub alpha: f64,
    /// `β` — how strongly failure and drift lower `Ω`.
    pub beta: f64,
    /// `γ` — emergence coefficient on the pairwise `A_i A_j W_ij` term.
    pub gamma: f64,
    /// `λ` — trust growth per successful collaboration.
    ///
    /// `λ` and `μ` are best read together. Because `T_ij += λS − μE` is linear and
    /// clamped to `[0, 1]`, trust is a *threshold classifier*: a partner whose success
    /// rate exceeds `μ / (λ + μ)` drifts to full trust, and one below it drifts to zero.
    /// That break-even rate — not the individual values — is the knob that decides who
    /// the mesh ends up routing to.
    pub lambda: f64,
    /// `μ` — trust decay per execution failure. See [`MeshCoefficients::lambda`] for how
    /// the two together set the reliability bar.
    pub mu: f64,
    /// `η` — memory learning rate on acquired knowledge `K_i`.
    pub eta: f64,
    /// `δ` — forgetting rate on memory decay `R_i`.
    pub delta: f64,
    /// `σ` — confidence gain per success.
    pub sigma: f64,
    /// `ρ` — confidence loss per error.
    pub rho: f64,
    /// `κ` — specialization gain per unit of domain experience.
    pub kappa: f64,
    /// `ω` — specialization loss to obsolescence. Named in full to avoid colliding
    /// with `Ω`, the global intelligence state.
    pub obsolescence: f64,
    /// `φ` — plasticity gain: how much trust strengthens a connection.
    pub phi: f64,
    /// `ψ` — plasticity cost: how much latency weakens a connection.
    pub psi: f64,
    /// `τ` — priority denominator floor and urgency term. See
    /// [`crate::task::PriorityScore`] for why urgency also enters the numerator.
    pub tau: f64,
}

impl Default for MeshCoefficients {
    fn default() -> Self {
        Self {
            alpha: 0.35,
            beta: 0.20,
            gamma: 0.08,
            // Break-even reliability of 0.25 / (0.10 + 0.25) ≈ 71%: the mesh leans on
            // partners that clear roughly three jobs in four and writes off the rest.
            lambda: 0.10,
            mu: 0.25,
            eta: 0.18,
            delta: 0.04,
            sigma: 0.12,
            rho: 0.09,
            kappa: 0.10,
            obsolescence: 0.02,
            phi: 0.15,
            psi: 0.05,
            tau: 0.15,
        }
    }
}

impl MeshCoefficients {
    /// A conservative profile: slow to trust, slow to forget, hard to destabilize.
    ///
    /// Break-even reliability ≈ 83% — this mesh only leans on partners that almost
    /// never miss.
    pub fn conservative() -> Self {
        Self {
            alpha: 0.20,
            beta: 0.30,
            gamma: 0.04,
            lambda: 0.05,
            mu: 0.25,
            eta: 0.10,
            delta: 0.02,
            sigma: 0.06,
            rho: 0.14,
            kappa: 0.06,
            obsolescence: 0.01,
            phi: 0.08,
            psi: 0.09,
            tau: 0.25,
        }
    }

    /// An exploratory profile: fast emergence, fast trust, more churn.
    ///
    /// Break-even reliability ≈ 33% — this mesh keeps talking to partners that often
    /// fail, on the bet that the connection is worth more than the misses.
    pub fn exploratory() -> Self {
        Self {
            alpha: 0.50,
            beta: 0.15,
            gamma: 0.16,
            lambda: 0.30,
            mu: 0.15,
            eta: 0.28,
            delta: 0.08,
            sigma: 0.18,
            rho: 0.07,
            kappa: 0.16,
            obsolescence: 0.04,
            phi: 0.24,
            psi: 0.03,
            tau: 0.10,
        }
    }

    /// Reject coefficient sets that would make the mesh diverge or stall.
    ///
    /// The rules are deliberately loose — they catch nonsense (negative rates, a
    /// zero `τ` that divides by zero, an emergence term large enough to explode)
    /// rather than prescribing a single stable regime.
    pub fn validate(&self) -> Result<(), CoefficientError> {
        let rates = [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("lambda", self.lambda),
            ("mu", self.mu),
            ("eta", self.eta),
            ("delta", self.delta),
            ("sigma", self.sigma),
            ("rho", self.rho),
            ("kappa", self.kappa),
            ("obsolescence", self.obsolescence),
            ("phi", self.phi),
            ("psi", self.psi),
        ];

        for (name, value) in rates {
            if !value.is_finite() || value < 0.0 {
                return Err(CoefficientError::NonNegativeRate { name, value });
            }
            if value > 1.0 {
                return Err(CoefficientError::RateTooLarge { name, value });
            }
        }

        if !self.tau.is_finite() || self.tau <= 0.0 {
            return Err(CoefficientError::TauMustBePositive { value: self.tau });
        }

        if self.gamma > 0.5 {
            return Err(CoefficientError::EmergenceUnstable { gamma: self.gamma });
        }

        Ok(())
    }
}

/// Why a coefficient set was rejected.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CoefficientError {
    NonNegativeRate { name: &'static str, value: f64 },
    RateTooLarge { name: &'static str, value: f64 },
    TauMustBePositive { value: f64 },
    EmergenceUnstable { gamma: f64 },
}

impl std::fmt::Display for CoefficientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoefficientError::NonNegativeRate { name, value } => {
                write!(f, "coefficient `{name}` must be finite and non-negative, got {value}")
            }
            CoefficientError::RateTooLarge { name, value } => {
                write!(f, "coefficient `{name}` must be <= 1.0 to stay stable, got {value}")
            }
            CoefficientError::TauMustBePositive { value } => {
                write!(f, "tau must be > 0 so the priority denominator never vanishes, got {value}")
            }
            CoefficientError::EmergenceUnstable { gamma } => {
                write!(f, "gamma > 0.5 lets the pairwise emergence term dominate, got {gamma}")
            }
        }
    }
}

impl std::error::Error for CoefficientError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid() {
        MeshCoefficients::default().validate().unwrap();
        MeshCoefficients::conservative().validate().unwrap();
        MeshCoefficients::exploratory().validate().unwrap();
    }

    #[test]
    fn zero_tau_is_rejected() {
        let coefficients = MeshCoefficients {
            tau: 0.0,
            ..MeshCoefficients::default()
        };
        assert!(matches!(
            coefficients.validate(),
            Err(CoefficientError::TauMustBePositive { .. })
        ));
    }

    #[test]
    fn runaway_emergence_is_rejected() {
        let coefficients = MeshCoefficients {
            gamma: 0.9,
            ..MeshCoefficients::default()
        };
        assert!(matches!(
            coefficients.validate(),
            Err(CoefficientError::EmergenceUnstable { .. })
        ));
    }

    #[test]
    fn negative_rates_are_rejected() {
        let coefficients = MeshCoefficients {
            lambda: -0.1,
            ..MeshCoefficients::default()
        };
        assert!(matches!(
            coefficients.validate(),
            Err(CoefficientError::NonNegativeRate { name: "lambda", .. })
        ));
    }
}
