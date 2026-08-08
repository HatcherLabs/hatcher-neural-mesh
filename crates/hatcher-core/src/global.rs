//! Layer 1: the global intelligence state `Ω` and its ledger.
//!
//! `Ω_{t+1} = Ω_t + α(L + E + C) − β(F + D)`
//!
//! Every pipeline run produces exactly one [`OmegaDelta`] — the measured `L, E, C,
//! F, D` for that run — which is applied to [`GlobalState`] and appended to the
//! [`OmegaLedger`]. The ledger is the series the Analytics view plots.

use serde::{Deserialize, Serialize};

/// The measured inputs to one `Ω` update.
///
/// Each term is normalized to roughly `[0, 1]` per run so `α` and `β` stay
/// interpretable across cohorts of different sizes.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct OmegaDelta {
    /// `L` — learning gain: knowledge admitted to memory this run, `Σ η K_i`, normalized by cohort size.
    pub learning: f64,
    /// `E` — emergent behaviors discovered: the share of mesh intelligence coming
    /// from the pairwise term rather than from raw per-agent capability.
    pub emergence: f64,
    /// `C` — collaboration efficiency: trust-weighted successful handoffs over attempted handoffs.
    pub collaboration: f64,
    /// `F` — failure accumulation: failed stages over executed stages.
    pub failure: f64,
    /// `D` — drift from objectives: `1 − verifier confidence`, plus accumulated obsolescence.
    pub drift: f64,
}

impl OmegaDelta {
    /// `α(L + E + C)` — the gain side of the master equation.
    pub fn gain(&self, alpha: f64) -> f64 {
        alpha * (self.learning + self.emergence + self.collaboration)
    }

    /// `β(F + D)` — the penalty side of the master equation.
    pub fn penalty(&self, beta: f64) -> f64 {
        beta * (self.failure + self.drift)
    }

    /// Net movement in `Ω` this run.
    pub fn net(&self, alpha: f64, beta: f64) -> f64 {
        self.gain(alpha) - self.penalty(beta)
    }
}

/// The mesh's global intelligence state, plus the last measured terms.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct GlobalState {
    /// `Ω` — global mesh intelligence.
    pub omega: f64,
    /// The `L, E, C, F, D` measured on the most recent update.
    pub last: OmegaDelta,
    /// Number of updates applied so far.
    pub epoch: u64,
    /// Mesh intelligence `A` at the most recent update, for reference.
    pub aggregate_intelligence: f64,
}

impl Default for GlobalState {
    fn default() -> Self {
        Self {
            omega: 1.0,
            last: OmegaDelta::default(),
            epoch: 0,
            aggregate_intelligence: 0.0,
        }
    }
}

impl GlobalState {
    /// Start from an explicit `Ω`.
    pub fn starting_at(omega: f64) -> Self {
        Self {
            omega,
            ..Self::default()
        }
    }

    /// Regime label used by the UX and the Overview tab.
    pub fn regime(&self) -> OmegaRegime {
        let net = self.last.gain(1.0) - self.last.penalty(1.0);
        if self.omega < 0.5 {
            OmegaRegime::Degraded
        } else if net > 0.15 {
            OmegaRegime::Compounding
        } else if net < -0.15 {
            OmegaRegime::Eroding
        } else {
            OmegaRegime::Steady
        }
    }
}

/// Qualitative reading of where the mesh is heading.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OmegaRegime {
    /// `Ω` is rising: learning and collaboration outrun failure and drift.
    Compounding,
    /// `Ω` is roughly flat.
    Steady,
    /// `Ω` is falling: failure and drift dominate.
    Eroding,
    /// `Ω` has fallen far enough that the mesh should stop taking production work.
    Degraded,
}

impl OmegaRegime {
    pub fn as_str(&self) -> &'static str {
        match self {
            OmegaRegime::Compounding => "compounding",
            OmegaRegime::Steady => "steady",
            OmegaRegime::Eroding => "eroding",
            OmegaRegime::Degraded => "degraded",
        }
    }
}

/// One row of the `Ω` time series.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OmegaSample {
    pub epoch: u64,
    pub omega: f64,
    pub delta: OmegaDelta,
    pub aggregate_intelligence: f64,
    pub regime_gain: f64,
    pub regime_penalty: f64,
}

/// A bounded history of `Ω` updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmegaLedger {
    samples: Vec<OmegaSample>,
    capacity: usize,
}

impl Default for OmegaLedger {
    fn default() -> Self {
        Self::with_capacity(512)
    }
}

impl OmegaLedger {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    pub fn record(&mut self, sample: OmegaSample) {
        if self.samples.len() == self.capacity {
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    pub fn samples(&self) -> &[OmegaSample] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn latest(&self) -> Option<&OmegaSample> {
        self.samples.last()
    }

    /// Mean movement in `Ω` over the last `window` samples — the mesh's learning slope.
    pub fn slope(&self, window: usize) -> f64 {
        let window = window.max(2).min(self.samples.len());
        if window < 2 {
            return 0.0;
        }
        let tail = &self.samples[self.samples.len() - window..];
        let first = tail.first().map(|s| s.omega).unwrap_or(0.0);
        let last = tail.last().map(|s| s.omega).unwrap_or(0.0);
        (last - first) / (window as f64 - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_and_penalty_split_the_master_equation() {
        let delta = OmegaDelta {
            learning: 0.2,
            emergence: 0.1,
            collaboration: 0.3,
            failure: 0.1,
            drift: 0.2,
        };
        assert!((delta.gain(0.5) - 0.3).abs() < 1e-12);
        assert!((delta.penalty(0.5) - 0.15).abs() < 1e-12);
        assert!((delta.net(0.5, 0.5) - 0.15).abs() < 1e-12);
    }

    #[test]
    fn regime_reads_direction_then_absolute_health() {
        let mut state = GlobalState::default();
        state.last.learning = 0.4;
        assert_eq!(state.regime(), OmegaRegime::Compounding);

        state.last = OmegaDelta {
            failure: 0.5,
            drift: 0.3,
            ..OmegaDelta::default()
        };
        assert_eq!(state.regime(), OmegaRegime::Eroding);

        state.omega = 0.1;
        assert_eq!(state.regime(), OmegaRegime::Degraded, "a collapsed omega outranks direction");
    }

    #[test]
    fn ledger_is_bounded_and_reports_slope() {
        let mut ledger = OmegaLedger::with_capacity(3);
        for epoch in 1..=5 {
            ledger.record(OmegaSample {
                epoch,
                omega: epoch as f64,
                delta: OmegaDelta::default(),
                aggregate_intelligence: 0.0,
                regime_gain: 0.0,
                regime_penalty: 0.0,
            });
        }
        assert_eq!(ledger.len(), 3, "ledger must stay bounded");
        assert_eq!(ledger.latest().unwrap().epoch, 5);
        assert!((ledger.slope(3) - 1.0).abs() < 1e-12);
    }
}
