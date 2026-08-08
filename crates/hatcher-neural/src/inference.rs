//! The decision head: the actual neural network behind the mesh.
//!
//! The HAMNS equations govern the *graph* — who is capable, who is trusted, what
//! runs next. This module governs the *decision*: given a feature vector plus the
//! mesh's own summary statistics, which of the four control actions should the mesh
//! return to Hatcher?
//!
//! Two backends implement the same [`InferenceBackend`] contract:
//!
//! * [`NativeBackend`] — a small deterministic MLP built into the crate, so the mesh
//!   has no external dependency and every run is reproducible.
//! * `OnnxBackend` (see [`crate::onnx`]) — a real ONNX Runtime session, so a trained
//!   policy can be dropped in without touching the mesh.

use std::fmt;
use std::sync::Arc;

use hatcher_core::{MeshAction, ModelSpec};
use ndarray::{Array1, Array2};

use crate::equations::{deterministic_unit, seed_of};

/// Why an inference call failed.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    /// The input vector length did not match the model's declared input dimension.
    InputShape { expected: usize, actual: usize },
    /// The model returned an output vector of unexpected length.
    OutputShape { expected: usize, actual: usize },
    /// The model file could not be found.
    ModelNotFound { path: String },
    /// The backend was requested but the crate was built without its feature.
    BackendUnavailable { backend: &'static str, feature: &'static str },
    /// The runtime rejected the model or the call.
    Runtime { detail: String },
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferenceError::InputShape { expected, actual } => {
                write!(f, "model expects {expected} input features, got {actual}")
            }
            InferenceError::OutputShape { expected, actual } => {
                write!(f, "model should emit {expected} outputs, got {actual}")
            }
            InferenceError::ModelNotFound { path } => write!(f, "onnx model not found at `{path}`"),
            InferenceError::BackendUnavailable { backend, feature } => write!(
                f,
                "the `{backend}` backend requires building hatcher-neural with `--features {feature}`"
            ),
            InferenceError::Runtime { detail } => write!(f, "inference runtime error: {detail}"),
        }
    }
}

impl std::error::Error for InferenceError {}

/// A model that maps mesh features to action logits.
pub trait InferenceBackend: fmt::Debug + Send + Sync {
    /// Human-readable backend name, surfaced in traces and the Config tab.
    fn name(&self) -> &str;

    /// The shape contract this backend was loaded against.
    fn spec(&self) -> &ModelSpec;

    /// Run a forward pass. Output length must equal `spec().output_dim`.
    fn forward(&self, input: &[f64]) -> Result<Vec<f64>, InferenceError>;

    /// Pad or truncate an arbitrary feature vector to the model's input width.
    ///
    /// The mesh is fed by upstream systems that do not know the model's shape, so
    /// conforming here is preferable to rejecting live traffic.
    fn conform(&self, features: &[f64]) -> Vec<f64> {
        let width = self.spec().input_dim;
        let mut input = vec![0.0; width];
        for (slot, value) in input.iter_mut().zip(features.iter()) {
            *slot = if value.is_finite() { *value } else { 0.0 };
        }
        input
    }

    /// Forward pass plus argmax decoding into a control action.
    fn decide(&self, features: &[f64]) -> Result<Decision, InferenceError> {
        let input = self.conform(features);
        let logits = self.forward(&input)?;
        if logits.len() != self.spec().output_dim {
            return Err(InferenceError::OutputShape {
                expected: self.spec().output_dim,
                actual: logits.len(),
            });
        }
        Ok(Decision::from_distribution(&logits))
    }
}

/// A decoded decision: which action, how strongly, and the full distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub action: MeshAction,
    /// Probability mass on the chosen action.
    pub confidence: f64,
    /// The full action distribution, ordered as [`Decision::ACTION_ORDER`].
    pub distribution: Vec<f64>,
}

impl Decision {
    /// Output index → action. A four-wide head is one logit per action.
    pub const ACTION_ORDER: [MeshAction; 4] = [
        MeshAction::Observe,
        MeshAction::Delegate,
        MeshAction::Stabilize,
        MeshAction::Escalate,
    ];

    /// Decode a distribution (or raw logits) by argmax.
    ///
    /// Values that already look like a probability distribution are used as-is;
    /// anything else is softmaxed. Without that check, a backend that emits
    /// probabilities would be squashed a second time and every decision would come
    /// back with the same washed-out confidence.
    pub fn from_distribution(values: &[f64]) -> Self {
        let normalized = if is_distribution(values) {
            values.to_vec()
        } else {
            softmax(values)
        };
        let (index, confidence) = normalized
            .iter()
            .enumerate()
            .fold((0usize, f64::MIN), |acc, (index, value)| {
                if *value > acc.1 {
                    (index, *value)
                } else {
                    acc
                }
            });

        Self {
            action: Self::ACTION_ORDER
                .get(index)
                .copied()
                .unwrap_or(MeshAction::Observe),
            confidence: confidence.clamp(0.0, 1.0),
            distribution: normalized,
        }
    }
}

/// Whether a vector is already a probability distribution.
pub fn is_distribution(values: &[f64]) -> bool {
    !values.is_empty()
        && values.iter().all(|value| value.is_finite() && *value >= 0.0)
        && (values.iter().sum::<f64>() - 1.0).abs() < 1e-6
}

/// Numerically stable softmax over raw logits.
pub fn softmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().copied().filter(|v| v.is_finite()).fold(f64::MIN, f64::max);
    let max = if max.is_finite() { max } else { 0.0 };
    let exponentiated: Vec<f64> = values
        .iter()
        .map(|value| if value.is_finite() { (*value - max).exp() } else { 0.0 })
        .collect();
    let sum: f64 = exponentiated.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / values.len() as f64; values.len()];
    }
    exponentiated.into_iter().map(|value| value / sum).collect()
}

/// A small dense network: `input → tanh(hidden) → softmax(output)`.
///
/// Weights are derived deterministically from a seed string, so a fresh mesh on a
/// different machine produces byte-identical decisions. Call
/// [`NativeBackend::with_weights`] to install trained parameters instead.
#[derive(Debug, Clone)]
pub struct NativeBackend {
    spec: ModelSpec,
    hidden_weights: Array2<f64>,
    hidden_bias: Array1<f64>,
    output_weights: Array2<f64>,
    output_bias: Array1<f64>,
}

impl NativeBackend {
    /// Build a network with deterministic pseudo-random weights.
    pub fn new(spec: ModelSpec) -> Self {
        let seed = seed_of(&format!("{}:{}", spec.name, spec.version));
        let hidden_weights = Self::init_matrix(spec.hidden_dim, spec.input_dim, seed);
        let hidden_bias = Self::init_vector(spec.hidden_dim, seed ^ 0x11);
        let output_weights = Self::init_matrix(spec.output_dim, spec.hidden_dim, seed ^ 0x22);
        let output_bias = Self::init_vector(spec.output_dim, seed ^ 0x33);

        Self {
            spec,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
        }
    }

    /// Install explicit parameters, e.g. from an offline training run.
    pub fn with_weights(
        spec: ModelSpec,
        hidden_weights: Array2<f64>,
        hidden_bias: Array1<f64>,
        output_weights: Array2<f64>,
        output_bias: Array1<f64>,
    ) -> Result<Self, InferenceError> {
        if hidden_weights.shape() != [spec.hidden_dim, spec.input_dim] {
            return Err(InferenceError::Runtime {
                detail: format!(
                    "hidden weights must be {}x{}, got {:?}",
                    spec.hidden_dim, spec.input_dim, hidden_weights.shape()
                ),
            });
        }
        if output_weights.shape() != [spec.output_dim, spec.hidden_dim] {
            return Err(InferenceError::Runtime {
                detail: format!(
                    "output weights must be {}x{}, got {:?}",
                    spec.output_dim, spec.hidden_dim, output_weights.shape()
                ),
            });
        }

        Ok(Self {
            spec,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
        })
    }

    fn init_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        // Xavier-ish scaling keeps early activations off the saturating tails.
        let scale = (6.0 / (rows + cols).max(1) as f64).sqrt();
        Array2::from_shape_fn((rows, cols), |(row, col)| {
            let draw = deterministic_unit(seed ^ ((row as u64) << 32) ^ col as u64);
            (draw * 2.0 - 1.0) * scale
        })
    }

    fn init_vector(len: usize, seed: u64) -> Array1<f64> {
        Array1::from_shape_fn(len, |index| deterministic_unit(seed ^ index as u64) * 0.1 - 0.05)
    }
}

impl InferenceBackend for NativeBackend {
    fn name(&self) -> &str {
        "native"
    }

    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn forward(&self, input: &[f64]) -> Result<Vec<f64>, InferenceError> {
        if input.len() != self.spec.input_dim {
            return Err(InferenceError::InputShape {
                expected: self.spec.input_dim,
                actual: input.len(),
            });
        }

        let x = Array1::from_vec(input.to_vec());
        let hidden = (self.hidden_weights.dot(&x) + &self.hidden_bias).mapv(f64::tanh);
        let logits = self.output_weights.dot(&hidden) + &self.output_bias;
        Ok(softmax(logits.as_slice().unwrap_or(&[])))
    }
}

impl Default for NativeBackend {
    fn default() -> Self {
        Self::new(ModelSpec::native_default())
    }
}

/// Shared handle to a backend. `Arc` rather than `Box` so the mesh stays `Clone`
/// and a rehearsal can fork mesh state without reloading the model.
pub type SharedBackend = Arc<dyn InferenceBackend>;

/// Where a mesh should get its decision head from.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendKind {
    /// Use the built-in deterministic MLP.
    Native,
    /// Load an ONNX graph from disk.
    Onnx { model_path: String },
}

impl BackendKind {
    /// Resolve this choice into a live backend.
    ///
    /// An ONNX request on a build without the `onnx` feature — or with an
    /// unreadable model — is reported, never silently swapped. Callers that prefer
    /// degradation over failure should use [`resolve_or_native`].
    pub fn resolve(&self, spec: ModelSpec) -> Result<SharedBackend, InferenceError> {
        match self {
            BackendKind::Native => Ok(Arc::new(NativeBackend::new(spec))),
            BackendKind::Onnx { model_path } => {
                #[cfg(feature = "onnx")]
                {
                    Ok(Arc::new(crate::onnx::OnnxBackend::load(model_path, spec)?))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = (model_path, spec);
                    Err(InferenceError::BackendUnavailable {
                        backend: "onnx",
                        feature: "onnx",
                    })
                }
            }
        }
    }
}

/// Resolve a backend, falling back to the native head on failure.
///
/// Returns the backend plus the error that forced the fallback, if any, so the
/// caller can surface a degraded-mode warning instead of pretending all is well.
pub fn resolve_or_native(kind: &BackendKind, spec: ModelSpec) -> (SharedBackend, Option<InferenceError>) {
    match kind.resolve(spec.clone()) {
        Ok(backend) => (backend, None),
        Err(error) => (Arc::new(NativeBackend::new(spec)), Some(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full-width input for the canonical eight-slot layout.
    fn wide_input() -> Vec<f64> {
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }

    #[test]
    fn native_backend_emits_a_normalized_distribution() {
        let backend = NativeBackend::default();
        let output = backend.forward(&wide_input()).unwrap();
        assert_eq!(output.len(), 4);
        assert!((output.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(output.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn native_backend_is_deterministic_across_instances() {
        let left = NativeBackend::default().forward(&wide_input()).unwrap();
        let right = NativeBackend::default().forward(&wide_input()).unwrap();
        assert_eq!(left, right, "a fresh mesh must decide identically");
    }

    #[test]
    fn native_backend_responds_to_its_input() {
        let backend = NativeBackend::default();
        let calm = backend.forward(&vec![0.0; 8]).unwrap();
        let loud = backend.forward(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
        assert_ne!(calm, loud, "the head must actually depend on features");
    }

    #[test]
    fn wrong_input_width_is_rejected_but_conform_repairs_it() {
        let backend = NativeBackend::default();
        assert_eq!(
            backend.forward(&[0.1]),
            Err(InferenceError::InputShape { expected: 8, actual: 1 })
        );

        let conformed = backend.conform(&[0.1; 12]);
        assert_eq!(conformed.len(), 8, "extra features are truncated");
        assert_eq!(
            backend.conform(&[0.1, 0.2]),
            vec![0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "missing features pad with zero"
        );
        assert_eq!(backend.conform(&[f64::NAN, 0.2])[0], 0.0, "NaN is scrubbed");
    }

    #[test]
    fn decisions_decode_by_argmax_over_the_action_order() {
        let decision = Decision::from_distribution(&[0.0, 0.0, 9.0, 0.0]);
        assert_eq!(decision.action, MeshAction::Stabilize);
        assert!(decision.confidence > 0.9);
        assert_eq!(decision.distribution.len(), 4);

        let escalate = Decision::from_distribution(&[0.0, 0.0, 0.0, 9.0]);
        assert_eq!(escalate.action, MeshAction::Escalate);
    }

    #[test]
    fn softmax_survives_degenerate_input() {
        assert!(softmax(&[]).is_empty());
        let uniform = softmax(&[f64::NAN, f64::NAN]);
        assert!((uniform.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let big = softmax(&[1000.0, 1000.0]);
        assert!((big[0] - 0.5).abs() < 1e-9, "large logits must not overflow");
    }

    #[test]
    fn explicit_weights_are_shape_checked() {
        let spec = ModelSpec::native_default();
        let bad = NativeBackend::with_weights(
            spec.clone(),
            Array2::zeros((2, 2)),
            Array1::zeros(spec.hidden_dim),
            Array2::zeros((spec.output_dim, spec.hidden_dim)),
            Array1::zeros(spec.output_dim),
        );
        assert!(matches!(bad, Err(InferenceError::Runtime { .. })));

        let good = NativeBackend::with_weights(
            spec.clone(),
            Array2::zeros((spec.hidden_dim, spec.input_dim)),
            Array1::zeros(spec.hidden_dim),
            Array2::zeros((spec.output_dim, spec.hidden_dim)),
            Array1::zeros(spec.output_dim),
        );
        assert!(good.is_ok());
    }

    #[test]
    fn decide_runs_the_whole_path_from_raw_features() {
        let backend = NativeBackend::default();
        let decision = backend.decide(&[0.9]).unwrap();
        assert!(Decision::ACTION_ORDER.contains(&decision.action));
        assert!(decision.confidence > 0.0);
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn onnx_without_the_feature_reports_instead_of_pretending() {
        let kind = BackendKind::Onnx {
            model_path: "models/policy.onnx".into(),
        };
        assert!(matches!(
            kind.resolve(ModelSpec::native_default()),
            Err(InferenceError::BackendUnavailable {
                backend: "onnx",
                feature: "onnx"
            })
        ));

        let (backend, error) = resolve_or_native(&kind, ModelSpec::native_default());
        assert_eq!(backend.name(), "native", "fallback keeps the mesh running");
        assert!(error.is_some(), "but the degradation is reported");
    }
}
