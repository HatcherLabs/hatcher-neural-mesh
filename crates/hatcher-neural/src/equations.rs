//! The ten HAMNS update rules, as pure functions.
//!
//! Everything in this module is total, deterministic, and free of side effects: the
//! same inputs always give the same output, and nothing here reads mesh state it
//! was not handed. The stateful engine in [`crate::mesh`] and the pipeline in
//! [`crate::pipeline`] are the only things that decide *when* these fire and *what*
//! is measured; this module decides only *how* the numbers move.
//!
//! Each function names its equation in its doc comment, and each is bounded so a
//! misconfigured coefficient set degrades rather than diverges.

use hatcher_core::{
    unit, CapabilityVector, MeshCoefficients, MeshIntelligence, OmegaDelta, PriorityScore,
};
use ndarray::Array2;

/// Numerical floor for `Ω`. A mesh cannot have negative intelligence; it can only
/// be fully degraded.
pub const OMEGA_FLOOR: f64 = 0.0;

/// Numerical ceiling for `Ω`. Not a claim about intelligence, just a guard that
/// keeps a runaway `α` from producing `inf` and poisoning every downstream ratio.
pub const OMEGA_CEILING: f64 = 16.0;

// ---------------------------------------------------------------------------
// 1. Global intelligence state
// ---------------------------------------------------------------------------

/// `Ω(t+1) = Ω(t) + α(L + E + C) − β(F + D)`
///
/// Every cycle the mesh gains intelligence through learning, emergent discovery,
/// and collaboration, and loses it through failure and drift from objectives.
pub fn global_intelligence(omega: f64, delta: &OmegaDelta, coefficients: &MeshCoefficients) -> f64 {
    let next = omega + delta.net(coefficients.alpha, coefficients.beta);
    if next.is_nan() {
        return omega;
    }
    next.clamp(OMEGA_FLOOR, OMEGA_CEILING)
}

// ---------------------------------------------------------------------------
// 2. Agent capability
// ---------------------------------------------------------------------------

/// `A_i = I_i · S_i · P_i · C_i · M_i`
///
/// Multiplicative on purpose. Weaknesses matter: a brilliant model with poor memory
/// is weak, and a specialist with no context is weak.
pub fn agent_capability(capability: &CapabilityVector) -> f64 {
    capability.capability()
}

// ---------------------------------------------------------------------------
// 3. Mesh intelligence
// ---------------------------------------------------------------------------

/// `A = Σ_i A_i + γ Σ_{i≠j} A_i A_j W_ij`
///
/// The first term is raw compute — what the cohort is worth with the wires cut. The
/// second is collective intelligence: two good agents are worth more than their sum
/// once they are connected, and worth nothing extra while `W_ij` is zero.
///
/// `weights` must be at least `caps.len()` square; the diagonal is ignored.
pub fn mesh_intelligence(caps: &[f64], weights: &Array2<f64>, gamma: f64) -> MeshIntelligence {
    let raw: f64 = caps.iter().copied().filter(|value| value.is_finite()).sum();

    let n = caps.len().min(weights.nrows()).min(weights.ncols());
    let mut pairwise = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let contribution = caps[i] * caps[j] * weights[[i, j]];
            if contribution.is_finite() {
                pairwise += contribution;
            }
        }
    }

    let emergent = gamma * pairwise;
    MeshIntelligence {
        raw,
        emergent,
        total: raw + emergent,
    }
}

/// The pairs contributing the most collective intelligence, strongest first.
///
/// This is what makes emergence legible: it names the specific collaborations that
/// are producing more than the agents could alone.
pub fn top_emergent_pairs(caps: &[f64], weights: &Array2<f64>, gamma: f64, limit: usize) -> Vec<(usize, usize, f64)> {
    let n = caps.len().min(weights.nrows()).min(weights.ncols());
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let contribution = gamma * caps[i] * caps[j] * weights[[i, j]];
            if contribution.is_finite() && contribution > 0.0 {
                pairs.push((i, j, contribution));
            }
        }
    }
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(limit);
    pairs
}

// ---------------------------------------------------------------------------
// 4. Dynamic trust
// ---------------------------------------------------------------------------

/// `T_ij(t+1) = T_ij(t) + λ S_ij − μ E_ij`
///
/// Over time the mesh learns who should talk to whom: unreliable agents lose trust
/// until nothing routes to them, and reliable agents accumulate it until they are
/// hubs. Trust is held in `[0, 1]` so a long success streak cannot make an agent
/// unfalsifiably trusted.
pub fn trust_update(trust: f64, successes: f64, failures: f64, coefficients: &MeshCoefficients) -> f64 {
    let next = trust + coefficients.lambda * successes.max(0.0) - coefficients.mu * failures.max(0.0);
    unit(next)
}

// ---------------------------------------------------------------------------
// 5. Priority
// ---------------------------------------------------------------------------

/// `P = (U · B · I · urgency_gain) / (C + τ)`
///
/// High `P` means "uncertain, expensive, and the mesh is not confident — schedule it
/// now, on the strongest agents". Low `P` means "the mesh already knows this — defer
/// it, or hand it to a cheaper agent".
///
/// See [`PriorityScore`] for why urgency multiplies the numerator instead of joining
/// `τ` in the denominator. With `urgency_gain = 1.0` this is exactly the spec form.
pub fn priority(
    uncertainty: f64,
    budget: f64,
    implementation_cost: f64,
    confidence: f64,
    urgency_gain: f64,
    coefficients: &MeshCoefficients,
) -> PriorityScore {
    let uncertainty = unit(uncertainty);
    let budget = unit(budget);
    let implementation_cost = unit(implementation_cost);
    let confidence = unit(confidence);
    let tau = coefficients.tau.max(1e-6);
    let urgency_gain = urgency_gain.max(1.0);

    let numerator = uncertainty * budget * implementation_cost * urgency_gain;
    let value = numerator / (confidence + tau);

    PriorityScore {
        value,
        uncertainty,
        budget,
        implementation_cost,
        confidence,
        tau,
        urgency_gain,
        band: PriorityScore::band_for(value),
    }
}

/// Map a task's urgency in `[0, 1]` onto the numerator multiplier.
///
/// `urgency = 0` leaves priority untouched; `urgency = 1` doubles it.
pub fn urgency_gain(urgency: f64) -> f64 {
    1.0 + unit(urgency)
}

// ---------------------------------------------------------------------------
// 6. Memory evolution
// ---------------------------------------------------------------------------

/// `M_i(t+1) = M_i(t) + η K_i − δ R_i`
///
/// Memory quality rises with knowledge actually committed and falls with decay.
/// `K_i` and `R_i` are measured by the memory graph, not assumed.
pub fn memory_update(memory: f64, knowledge: f64, decay: f64, coefficients: &MeshCoefficients) -> f64 {
    unit(memory + coefficients.eta * knowledge.max(0.0) - coefficients.delta * decay.max(0.0))
}

// ---------------------------------------------------------------------------
// 7. Confidence calibration
// ---------------------------------------------------------------------------

/// `C_i(t+1) = C_i(t) + σ·success − ρ·error`
///
/// Lets an agent calibrate itself against outcomes instead of asserting certainty.
/// `success` and `error` are counts (or fractional evidence) from the run.
pub fn confidence_update(confidence: f64, success: f64, error: f64, coefficients: &MeshCoefficients) -> f64 {
    unit(confidence + coefficients.sigma * success.max(0.0) - coefficients.rho * error.max(0.0))
}

// ---------------------------------------------------------------------------
// 8. Specialization evolution
// ---------------------------------------------------------------------------

/// `S_i(t+1) = S_i(t) + κ·experience − ω·obsolescence`
///
/// Agents become experts in the domains they repeatedly solve, and lose edge in
/// domains that move on without them.
pub fn specialization_update(
    specialization: f64,
    experience: f64,
    obsolescence: f64,
    coefficients: &MeshCoefficients,
) -> f64 {
    unit(specialization + coefficients.kappa * experience.max(0.0) - coefficients.obsolescence * obsolescence.max(0.0))
}

// ---------------------------------------------------------------------------
// 9. Network plasticity
// ---------------------------------------------------------------------------

/// `W_ij(t+1) = W_ij(t) + φ T_ij (1 − W_ij) − ψ·latency`
///
/// Connections strengthen through trust and weaken under communication cost, so the
/// mesh does not just learn who is good — it learns who is worth the round trip.
///
/// The `(1 − W_ij)` factor is the one place this implementation refines the stated
/// rule, and it earns its place. Written literally as `W += φT − ψ·latency`, the
/// update is linear and unbounded: any link whose trust clears `ψ·latency / φ` grows
/// without limit until it hits the `[0, 1]` clamp, so in practice *every* link that
/// ever carries traffic pins at `1.0` and `W` stops distinguishing a strong partner
/// from a tolerable one. Making the bound smooth instead of a wall gives the equation
/// an interior fixed point,
///
/// ```text
/// W* = 1 − (ψ·latency) / (φ·T)
/// ```
///
/// so a link settles at a strength that actually reflects its trust-to-cost ratio,
/// and a link whose trust falls below `ψ·latency / φ` dies off on its own.
pub fn plasticity_update(weight: f64, trust: f64, latency: f64, coefficients: &MeshCoefficients) -> f64 {
    let weight = unit(weight);
    let gain = coefficients.phi * unit(trust) * (1.0 - weight);
    unit(weight + gain - coefficients.psi * latency.max(0.0))
}

/// The strength a link settles at, given steady trust and latency.
///
/// Returns `0.0` for links that cannot pay for themselves. Useful for explaining why
/// the mesh dropped a connection without having to replay the whole history.
pub fn plasticity_fixed_point(trust: f64, latency: f64, coefficients: &MeshCoefficients) -> f64 {
    let gain = coefficients.phi * unit(trust);
    if gain <= 0.0 {
        return 0.0;
    }
    unit(1.0 - (coefficients.psi * latency.max(0.0)) / gain)
}

// ---------------------------------------------------------------------------
// 10. Resource ratio
// ---------------------------------------------------------------------------

/// `R_i = P_i / (Energy_i + Latency_i)`
///
/// Performance per unit of cost — the tiebreaker when several agents can do the job.
pub fn resource_efficiency(performance: f64, energy: f64, latency: f64) -> f64 {
    let cost = (energy.max(0.0) + latency.max(0.0)).max(1e-6);
    (performance.max(0.0) / cost).max(0.0)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Squashing activation used by mesh message passing, mapping `ℝ → (0, 1)`.
pub fn squash(value: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    1.0 / (1.0 + (-value).exp())
}

/// Deterministic pseudo-randomness. The mesh must be reproducible: given the same
/// task id and the same state, a rehearsal has to replay identically, so outcome
/// sampling is seeded by content rather than by a clock.
pub fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic draw in `[0, 1)` from a seed.
pub fn deterministic_unit(seed: u64) -> f64 {
    (splitmix64(seed) >> 11) as f64 / (1u64 << 53) as f64
}

/// Stable seed for a string, so task ids and agent ids drive reproducible draws.
pub fn seed_of(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::PriorityBand;
    use ndarray::array;

    fn coefficients() -> MeshCoefficients {
        MeshCoefficients::default()
    }

    #[test]
    fn omega_rises_with_learning_and_falls_with_failure() {
        let coefficients = coefficients();
        let learning = OmegaDelta {
            learning: 0.4,
            emergence: 0.2,
            collaboration: 0.3,
            ..OmegaDelta::default()
        };
        let failing = OmegaDelta {
            failure: 0.8,
            drift: 0.5,
            ..OmegaDelta::default()
        };

        assert!(global_intelligence(1.0, &learning, &coefficients) > 1.0);
        assert!(global_intelligence(1.0, &failing, &coefficients) < 1.0);
    }

    #[test]
    fn omega_is_bounded_below_at_zero_and_above_at_the_ceiling() {
        let coefficients = coefficients();
        let collapse = OmegaDelta {
            failure: 1.0,
            drift: 1.0,
            ..OmegaDelta::default()
        };
        assert_eq!(global_intelligence(0.05, &collapse, &coefficients), OMEGA_FLOOR);
        let boom = OmegaDelta {
            learning: 1.0,
            emergence: 1.0,
            collaboration: 1.0,
            ..OmegaDelta::default()
        };
        assert_eq!(global_intelligence(OMEGA_CEILING, &boom, &coefficients), OMEGA_CEILING);
    }

    #[test]
    fn mesh_intelligence_splits_raw_compute_from_emergence() {
        let caps = vec![0.5, 0.5];
        let disconnected = Array2::<f64>::zeros((2, 2));
        let connected = array![[0.0, 1.0], [1.0, 0.0]];

        let alone = mesh_intelligence(&caps, &disconnected, 0.5);
        assert_eq!(alone.emergent, 0.0);
        assert_eq!(alone.total, 1.0);

        let together = mesh_intelligence(&caps, &connected, 0.5);
        // Two directed pairs, each 0.5 * 0.5 * 1.0, scaled by gamma = 0.5.
        assert!((together.emergent - 0.25).abs() < 1e-12);
        assert!(together.total > alone.total, "connection must beat isolation");
        assert!(together.emergence_ratio() > 0.0);
    }

    #[test]
    fn mesh_intelligence_ignores_the_diagonal_and_extra_matrix_rows() {
        let caps = vec![1.0, 1.0];
        let self_loops = array![[9.0, 0.0], [0.0, 9.0]];
        assert_eq!(mesh_intelligence(&caps, &self_loops, 1.0).emergent, 0.0);

        let oversized = Array2::<f64>::ones((5, 5));
        let bounded = mesh_intelligence(&caps, &oversized, 1.0);
        assert!((bounded.emergent - 2.0).abs() < 1e-12, "only the first two rows apply");
    }

    #[test]
    fn top_emergent_pairs_ranks_the_strongest_collaborations() {
        let caps = vec![0.9, 0.9, 0.1];
        let weights = array![[0.0, 1.0, 0.1], [1.0, 0.0, 0.1], [0.1, 0.1, 0.0]];
        let pairs = top_emergent_pairs(&caps, &weights, 0.1, 2);
        assert_eq!(pairs.len(), 2);
        assert!(pairs[0].2 >= pairs[1].2);
        assert!(matches!(pairs[0], (0, 1, _) | (1, 0, _)), "the two strong agents lead");
    }

    #[test]
    fn trust_grows_on_success_and_decays_on_failure() {
        let coefficients = coefficients();
        let grown = trust_update(0.5, 2.0, 0.0, &coefficients);
        let decayed = trust_update(0.5, 0.0, 2.0, &coefficients);
        assert!(grown > 0.5);
        assert!(decayed < 0.5);
        assert_eq!(trust_update(1.0, 100.0, 0.0, &coefficients), 1.0, "trust saturates");
        assert_eq!(trust_update(0.0, 0.0, 100.0, &coefficients), 0.0, "trust floors");
    }

    #[test]
    fn priority_rises_with_uncertainty_and_falls_with_confidence() {
        let coefficients = coefficients();
        let unknown = priority(0.9, 0.9, 0.9, 0.1, 1.0, &coefficients);
        let known = priority(0.9, 0.9, 0.9, 0.95, 1.0, &coefficients);
        assert!(unknown.value > known.value);
        assert_eq!(unknown.band, PriorityBand::Immediate);

        let trivial = priority(0.1, 0.1, 0.1, 0.9, 1.0, &coefficients);
        assert_eq!(trivial.band, PriorityBand::Deferred);
    }

    #[test]
    fn urgency_multiplies_priority_instead_of_suppressing_it() {
        let coefficients = coefficients();
        let calm = priority(0.6, 0.6, 0.6, 0.4, urgency_gain(0.0), &coefficients);
        let urgent = priority(0.6, 0.6, 0.6, 0.4, urgency_gain(1.0), &coefficients);
        assert!((urgent.value - calm.value * 2.0).abs() < 1e-12);
        assert_eq!(urgency_gain(0.5), 1.5);
    }

    #[test]
    fn priority_denominator_never_vanishes() {
        let mut coefficients = coefficients();
        coefficients.tau = 0.0;
        let score = priority(1.0, 1.0, 1.0, 0.0, 1.0, &coefficients);
        assert!(score.value.is_finite(), "tau is floored so P stays finite");
    }

    #[test]
    fn memory_learns_then_forgets() {
        let coefficients = coefficients();
        let learned = memory_update(0.5, 1.0, 0.0, &coefficients);
        assert!(learned > 0.5);
        let forgotten = memory_update(0.5, 0.0, 1.0, &coefficients);
        assert!(forgotten < 0.5);
        assert!((0.0..=1.0).contains(&memory_update(0.99, 10.0, 0.0, &coefficients)));
    }

    #[test]
    fn confidence_calibrates_in_both_directions() {
        let coefficients = coefficients();
        assert!(confidence_update(0.5, 1.0, 0.0, &coefficients) > 0.5);
        assert!(confidence_update(0.5, 0.0, 1.0, &coefficients) < 0.5);
    }

    #[test]
    fn specialization_accretes_with_experience() {
        let coefficients = coefficients();
        let mut specialization = 0.3;
        for _ in 0..5 {
            specialization = specialization_update(specialization, 1.0, 0.0, &coefficients);
        }
        assert!(specialization > 0.3, "repeated domain work compounds mastery");
        assert!(specialization_update(0.5, 0.0, 5.0, &coefficients) < 0.5);
    }

    #[test]
    fn plasticity_follows_trust_but_pays_for_latency() {
        let coefficients = coefficients();
        let near = plasticity_update(0.3, 0.9, 0.0, &coefficients);
        let far = plasticity_update(0.3, 0.9, 1.0, &coefficients);
        assert!(near > far, "a distant partner is a weaker connection at equal trust");
        assert!(plasticity_update(0.3, 0.0, 1.0, &coefficients) < 0.3);
    }

    #[test]
    fn plasticity_settles_instead_of_pinning_at_one() {
        let coefficients = coefficients();
        let mut trusted = 0.1;
        let mut tolerated = 0.1;
        for _ in 0..200 {
            trusted = plasticity_update(trusted, 1.0, 0.25, &coefficients);
            tolerated = plasticity_update(tolerated, 0.25, 0.25, &coefficients);
        }

        assert!(trusted > tolerated, "W must grade partners, not saturate for all of them");
        assert!(tolerated < 0.95, "a merely tolerated link should not pin at full strength");
        assert!((trusted - plasticity_fixed_point(1.0, 0.25, &coefficients)).abs() < 0.02);
        assert!((tolerated - plasticity_fixed_point(0.25, 0.25, &coefficients)).abs() < 0.02);
    }

    #[test]
    fn a_link_that_cannot_pay_for_itself_dies() {
        let coefficients = coefficients();
        assert_eq!(plasticity_fixed_point(0.0, 0.5, &coefficients), 0.0);
        assert_eq!(plasticity_fixed_point(0.02, 1.0, &coefficients), 0.0);

        let mut weight = 0.6;
        for _ in 0..200 {
            weight = plasticity_update(weight, 0.02, 1.0, &coefficients);
        }
        assert_eq!(weight, 0.0, "the mesh drops connections it cannot justify");
    }

    #[test]
    fn resource_efficiency_is_performance_per_cost() {
        assert!((resource_efficiency(0.8, 0.2, 0.2) - 2.0).abs() < 1e-12);
        assert!(resource_efficiency(0.8, 0.0, 0.0).is_finite(), "zero cost must not divide by zero");
    }

    #[test]
    fn squash_maps_into_the_unit_interval() {
        assert!((squash(0.0) - 0.5).abs() < 1e-12);
        assert!((0.99..=1.0).contains(&squash(50.0)), "large input saturates high");
        assert!((0.0..0.01).contains(&squash(-50.0)), "large negative input saturates low");
        assert_eq!(squash(f64::NAN), 0.0);
    }

    #[test]
    fn deterministic_draws_are_reproducible_and_well_distributed() {
        assert_eq!(deterministic_unit(seed_of("task-1")), deterministic_unit(seed_of("task-1")));
        assert_ne!(deterministic_unit(seed_of("task-1")), deterministic_unit(seed_of("task-2")));

        let mean = (0..1000).map(deterministic_unit).sum::<f64>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.05, "draws should be roughly uniform, got mean {mean}");
        assert!((0..1000).all(|i| (0.0..1.0).contains(&deterministic_unit(i))));
    }
}
