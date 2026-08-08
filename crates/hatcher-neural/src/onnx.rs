//! ONNX decision head.
//!
//! Enable with `--features onnx`. A trained policy replaces the built-in MLP without
//! any other change to the mesh: the equations, routing, and trust dynamics are
//! untouched, and only the mapping from mesh state to control action moves.
//!
//! ## Model contract
//!
//! ```text
//! input    float32   [1, spec.input_dim]     default [1, 8]
//! output   float32   [1, spec.output_dim]    default [1, 4]
//! ```
//!
//! Input slots are the canonical mesh feature layout documented on
//! [`crate::mesh::NeuralMesh::model_input`]; output slots are one score per action in
//! [`Decision::ACTION_ORDER`] — observe, delegate, stabilize, escalate. Either raw
//! logits or an already-normalized distribution is accepted; the decoder detects which.
//!
//! `scripts/export_policy_onnx.py` in this repository exports a conforming model and
//! documents the slots, and `models/fixtures/hamns-policy-8x4.onnx` is a small working
//! example used by the tests here.
//!
//! ## Runtime
//!
//! Inference runs on [tract](https://github.com/sonos/tract), a pure-Rust engine, so
//! enabling this feature does not introduce a native ONNX Runtime shared library that
//! has to be shipped alongside the binary and kept version-matched with the host. The
//! tradeoff is operator coverage: tract implements a large but not exhaustive subset of
//! ONNX, so an exotic graph may fail to load. That failure is reported at load time
//! with the operator named, and [`crate::inference::resolve_or_native`] will fall back
//! to the built-in head rather than leaving the mesh unable to decide anything.

use std::path::Path;
use std::sync::Arc;

use hatcher_core::ModelSpec;
use tract_onnx::prelude::*;

use crate::inference::{Decision, InferenceBackend, InferenceError};

/// An optimized, runnable ONNX graph bound to a mesh feature contract.
pub struct OnnxBackend {
    spec: ModelSpec,
    path: String,
    name: String,
    /// `SimplePlan::run` takes `&Arc<Self>`, so the plan is held behind an `Arc`.
    model: Arc<TypedRunnableModel>,
}

impl std::fmt::Debug for OnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The plan itself is enormous and not useful in a log line.
        f.debug_struct("OnnxBackend")
            .field("path", &self.path)
            .field("input_dim", &self.spec.input_dim)
            .field("output_dim", &self.spec.output_dim)
            .finish()
    }
}

fn runtime_error(error: impl std::fmt::Display) -> InferenceError {
    InferenceError::Runtime {
        detail: error.to_string(),
    }
}

impl OnnxBackend {
    /// Load, shape-bind, and optimize a model, then prove it runs.
    ///
    /// A probe pass with a zero vector happens at load time on purpose. Shape and
    /// operator problems otherwise surface on the first real request, which is the
    /// worst possible moment to discover that the policy was never viable.
    pub fn load(path: &str, spec: ModelSpec) -> Result<Self, InferenceError> {
        if !Path::new(path).exists() {
            return Err(InferenceError::ModelNotFound {
                path: path.to_string(),
            });
        }

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(runtime_error)?
            .with_input_fact(0, f32::fact([1, spec.input_dim]).into())
            .map_err(runtime_error)?
            .into_optimized()
            .map_err(runtime_error)?
            .into_runnable()
            .map_err(runtime_error)?;

        let backend = Self {
            name: format!("onnx:{}", file_stem(path)),
            spec,
            path: path.to_string(),
            model,
        };

        let probe = backend.forward(&vec![0.0; backend.spec.input_dim])?;
        if probe.len() != backend.spec.output_dim {
            return Err(InferenceError::OutputShape {
                expected: backend.spec.output_dim,
                actual: probe.len(),
            });
        }

        Ok(backend)
    }

    /// The file this policy was loaded from.
    pub fn path(&self) -> &str {
        &self.path
    }
}

fn file_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string())
}

impl InferenceBackend for OnnxBackend {
    fn name(&self) -> &str {
        &self.name
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

        // ONNX graphs are f32 by convention; the mesh works in f64. Narrowing here is
        // the only precision loss in the path, and it is bounded by the fact that every
        // input slot is already a normalized statistic in [0, 1].
        let values: Vec<f32> = input.iter().map(|value| *value as f32).collect();
        let tensor: Tensor = tract_ndarray::Array2::from_shape_vec((1, self.spec.input_dim), values)
            .map_err(runtime_error)?
            .into();

        let outputs = self.model.run(tvec!(tensor.into())).map_err(runtime_error)?;
        let first = outputs.first().ok_or_else(|| InferenceError::Runtime {
            detail: "model produced no outputs".to_string(),
        })?;
        let view = first.to_plain_array_view::<f32>().map_err(runtime_error)?;

        Ok(view.iter().map(|value| *value as f64).collect())
    }
}

/// Decode a raw model output into a control action, for callers holding logits directly.
pub fn decode(values: &[f64]) -> Decision {
    Decision::from_distribution(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{resolve_or_native, BackendKind};
    use crate::mesh::NeuralMesh;
    use hatcher_core::{MeshAction, TaskSpec};

    /// The checked-in policy exported by `scripts/export_policy_onnx.py`.
    const FIXTURE: &str = "models/fixtures/hamns-policy-8x4.onnx";

    /// Resolve the fixture against the repository root.
    ///
    /// The model lives in the repository, not inside this package, so it is absent when
    /// the crate is built from the published tarball. These tests skip loudly in that
    /// case rather than failing: the fixture's absence says nothing about the code.
    fn fixture() -> Option<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(FIXTURE);
        path.exists().then(|| path.to_string_lossy().into_owned())
    }

    /// Bind the fixture path or skip the test with an explanation.
    macro_rules! fixture_or_skip {
        () => {
            match fixture() {
                Some(path) => path,
                None => {
                    eprintln!("skipping: `{FIXTURE}` ships with the repository, not the published crate");
                    return;
                }
            }
        };
    }

    fn backend() -> Option<OnnxBackend> {
        let path = fixture()?;
        Some(OnnxBackend::load(&path, ModelSpec::native_default()).expect("fixture model should load"))
    }

    #[test]
    fn a_real_model_file_loads_and_runs() {
        let _ = fixture_or_skip!();
        let backend = backend().expect("fixture checked above");
        assert_eq!(backend.name(), "onnx:hamns-policy-8x4");
        assert_eq!(backend.spec().input_dim, 8);
        assert!(backend.path().ends_with(".onnx"));

        let output = backend.forward(&[0.5, 0.3, 0.8, 0.9, 0.3, 0.1, 0.2, 0.1]).unwrap();
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn the_policy_escalates_urgent_work_a_shaky_mesh_cannot_justify() {
        let _ = fixture_or_skip!();
        let decision = backend()
            .expect("fixture checked above")
            .decide(&[0.2, 0.1, 0.4, 0.15, 0.90, 0.9, 0.9, 0.9])
            .unwrap();
        assert_eq!(decision.action, MeshAction::Escalate);
        assert!(decision.confidence > 0.25);
    }

    #[test]
    fn the_policy_stabilizes_routine_work_a_confident_mesh_owns() {
        let _ = fixture_or_skip!();
        let decision = backend()
            .expect("fixture checked above")
            .decide(&[0.5, 0.3, 0.8, 0.9, 0.30, 0.1, 0.2, 0.1])
            .unwrap();
        assert_eq!(decision.action, MeshAction::Stabilize);
    }

    #[test]
    fn the_policy_delegates_when_the_cohort_is_trusted_but_unsure() {
        let _ = fixture_or_skip!();
        let decision = backend()
            .expect("fixture checked above")
            .decide(&[0.4, 0.7, 0.9, 0.45, 0.40, 0.4, 0.4, 0.4])
            .unwrap();
        assert_eq!(decision.action, MeshAction::Delegate);
    }

    #[test]
    fn inference_is_deterministic() {
        let _ = fixture_or_skip!();
        let features = [0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2];
        let backend = backend().expect("fixture checked above");
        assert_eq!(backend.forward(&features).unwrap(), backend.forward(&features).unwrap());
    }

    #[test]
    fn a_missing_model_is_reported_not_guessed_at() {
        let error = OnnxBackend::load("models/fixtures/does-not-exist.onnx", ModelSpec::native_default());
        assert!(matches!(error, Err(InferenceError::ModelNotFound { .. })));
    }

    #[test]
    fn a_file_that_is_not_a_model_fails_at_load_time() {
        let error = OnnxBackend::load("Cargo.toml", ModelSpec::native_default());
        assert!(
            matches!(error, Err(InferenceError::Runtime { .. })),
            "a non-model file must fail loudly at load, not on the first request"
        );
    }

    #[test]
    fn a_mismatched_input_contract_is_rejected_at_load_time() {
        // The fixture is an 8-wide model; binding it to a 4-wide contract must fail.
        let fixture = fixture_or_skip!();
        let error = OnnxBackend::load(&fixture, ModelSpec::new("mismatch", "1.0.0", 4, 6, 4));
        assert!(error.is_err(), "shape mismatches must not be discovered in production");
    }

    #[test]
    fn wrong_width_input_is_rejected_at_call_time() {
        let _ = fixture_or_skip!();
        assert_eq!(
            backend().expect("fixture checked above").forward(&[0.1, 0.2]),
            Err(InferenceError::InputShape { expected: 8, actual: 2 })
        );
    }

    #[test]
    fn a_mesh_can_run_its_decisions_through_the_onnx_head() {
        let fixture = fixture_or_skip!();
        let mesh = NeuralMesh::default()
            .try_with_onnx(&fixture)
            .expect("the mesh should accept a conforming policy");

        assert_eq!(mesh.backend().name(), "onnx:hamns-policy-8x4");
        assert!(mesh.degraded().is_none());

        let task = TaskSpec::new("task-1", "ship it", "rust")
            .with_features(vec![0.3, 0.7])
            .parsed_from_features();
        let priority = mesh.priority_for(&task);
        let decision = mesh.decide(&task, &priority).unwrap();
        assert!(Decision::ACTION_ORDER.contains(&decision.action));
    }

    #[test]
    fn the_onnx_head_actually_changes_what_the_mesh_decides() {
        let fixture = fixture_or_skip!();
        // A cold, unconfident mesh facing urgent, uncertain work.
        let task = {
            let mut task = TaskSpec::new("crisis", "everything is on fire", "rust");
            task.uncertainty = 1.0;
            task.budget = 1.0;
            task.implementation_cost = 1.0;
            task.urgency = 1.0;
            task
        };

        let onnx_mesh = NeuralMesh::default().try_with_onnx(&fixture).unwrap();
        let priority = onnx_mesh.priority_for(&task);
        let policy_decision = onnx_mesh.decide(&task, &priority).unwrap();

        assert_eq!(
            policy_decision.action,
            MeshAction::Escalate,
            "the trained policy should recognize a crisis"
        );
    }

    #[test]
    fn backend_kind_resolves_onnx_when_the_feature_is_on() {
        let fixture = fixture_or_skip!();
        let kind = BackendKind::Onnx { model_path: fixture };
        let backend = kind.resolve(ModelSpec::native_default()).unwrap();
        assert!(backend.name().starts_with("onnx:"));

        let (fallback, error) = resolve_or_native(
            &BackendKind::Onnx {
                model_path: "models/fixtures/missing.onnx".into(),
            },
            ModelSpec::native_default(),
        );
        assert_eq!(fallback.name(), "native", "a bad path degrades rather than dying");
        assert!(error.is_some());
    }

    #[test]
    fn decode_maps_logits_onto_the_action_order() {
        assert_eq!(decode(&[0.0, 0.0, 0.0, 5.0]).action, MeshAction::Escalate);
    }
}
