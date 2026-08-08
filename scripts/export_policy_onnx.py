#!/usr/bin/env python3
"""Export a HAMNS decision-head policy to ONNX.

The mesh's decision head is a small network that maps the canonical mesh feature
vector to one logit per control action. Any ONNX graph that honours the contract
below can be loaded by `hatcher-neural` with the `onnx` feature enabled:

    input   "mesh_features"   float32   [1, 8]
    output  "action_logits"   float32   [1, 4]

Input slots, in order:

    0  omega              global intelligence, normalized by OMEGA_CEILING (16.0)
    1  emergence_ratio    share of mesh intelligence A coming from connection
    2  mean_trust         mean T_ij across the cohort
    3  mean_confidence    mean calibrated agent confidence
    4  priority           P, squashed into (0, 1) by a logistic
    5  uncertainty        task U
    6  budget             task B
    7  implementation     task I

Output slots, in order: observe, delegate, stabilize, escalate.

The weights here are hand-set rather than trained, so the exported policy is
interpretable and testable: it escalates work that is high-priority while the mesh
is not confident, stabilizes work the mesh is confident and coherent about,
delegates when trust is high but confidence is middling, and observes otherwise.
Replace `policy_weights()` with trained parameters to ship a learned policy; the
contract and the export path stay the same.

Usage:
    python scripts/export_policy_onnx.py [--out models/fixtures/hamns-policy-8x4.onnx]
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 8
HIDDEN_DIM = 12
OUTPUT_DIM = 4

SLOT = {
    "omega": 0,
    "emergence": 1,
    "trust": 2,
    "confidence": 3,
    "priority": 4,
    "uncertainty": 5,
    "budget": 6,
    "implementation": 7,
}

ACTIONS = ["observe", "delegate", "stabilize", "escalate"]


class PolicyHead(nn.Module):
    """input -> tanh(hidden) -> logits, matching ModelSpec::native_default()."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, mesh_features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(mesh_features)))


def policy_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hand-set parameters implementing the interpretable policy described above.

    The hidden layer is a pass-through: the first eight units copy the input slots
    (scaled up so tanh stays in its near-linear region), and the last four are unused.
    All the policy logic then lives in the readable output layer.
    """
    hidden_w = np.zeros((HIDDEN_DIM, INPUT_DIM), dtype=np.float32)
    for slot in range(INPUT_DIM):
        hidden_w[slot, slot] = 1.0
    hidden_b = np.zeros(HIDDEN_DIM, dtype=np.float32)

    output_w = np.zeros((OUTPUT_DIM, HIDDEN_DIM), dtype=np.float32)
    output_b = np.zeros(OUTPUT_DIM, dtype=np.float32)

    observe, delegate, stabilize, escalate = range(OUTPUT_DIM)

    # Observe is the default floor: nothing stands out.
    output_b[observe] = 0.60

    # Delegate when the cohort is well connected and trusted, but not yet sure.
    output_w[delegate, SLOT["trust"]] = 2.20
    output_w[delegate, SLOT["emergence"]] = 1.60
    output_w[delegate, SLOT["confidence"]] = -0.80
    output_b[delegate] = -0.40

    # Stabilize when the mesh is confident and Omega is healthy, unless priority is high.
    output_w[stabilize, SLOT["confidence"]] = 3.20
    output_w[stabilize, SLOT["omega"]] = 2.00
    output_w[stabilize, SLOT["priority"]] = -2.40
    output_b[stabilize] = -0.60

    # Escalate when priority and uncertainty are high while confidence is not.
    output_w[escalate, SLOT["priority"]] = 4.20
    output_w[escalate, SLOT["uncertainty"]] = 2.00
    output_w[escalate, SLOT["implementation"]] = 0.80
    output_w[escalate, SLOT["confidence"]] = -3.20
    output_b[escalate] = -1.80

    return hidden_w, hidden_b, output_w, output_b


def build_model() -> PolicyHead:
    model = PolicyHead()
    hidden_w, hidden_b, output_w, output_b = policy_weights()
    with torch.no_grad():
        model.hidden.weight.copy_(torch.from_numpy(hidden_w))
        model.hidden.bias.copy_(torch.from_numpy(hidden_b))
        model.output.weight.copy_(torch.from_numpy(output_w))
        model.output.bias.copy_(torch.from_numpy(output_b))
    model.eval()
    return model


def describe(model: PolicyHead) -> None:
    """Print the decision for a few representative mesh states."""
    cases = {
        "confident mesh, routine task": [0.5, 0.3, 0.8, 0.9, 0.30, 0.1, 0.2, 0.1],
        "unsure mesh, urgent task": [0.2, 0.1, 0.4, 0.15, 0.90, 0.9, 0.9, 0.9],
        "well connected, middling confidence": [0.4, 0.7, 0.9, 0.45, 0.40, 0.4, 0.4, 0.4],
        "cold start": [0.06, 0.0, 0.5, 0.5, 0.35, 0.3, 0.3, 0.3],
    }
    print("\nsanity check:")
    for label, features in cases.items():
        with torch.no_grad():
            logits = model(torch.tensor([features], dtype=torch.float32))[0].numpy()
        print(f"  {label:<38} -> {ACTIONS[int(np.argmax(logits))]:<10} {np.round(logits, 3)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default="models/fixtures/hamns-policy-8x4.onnx",
        help="destination .onnx path",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    arguments = parser.parse_args()

    destination = pathlib.Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    sample = torch.zeros(1, INPUT_DIM, dtype=torch.float32)

    torch.onnx.export(
        model,
        (sample,),
        str(destination),
        input_names=["mesh_features"],
        output_names=["action_logits"],
        opset_version=arguments.opset,
        dynamo=False,
    )

    print(f"wrote {destination} ({destination.stat().st_size} bytes, opset {arguments.opset})")
    describe(model)


if __name__ == "__main__":
    main()
