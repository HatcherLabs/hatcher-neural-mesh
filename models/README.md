# Models

Decision-head policies for the mesh. See [`../docs/onnx.md`](../docs/onnx.md) for the
input/output contract and how loading works.

## Layout

| Path | Tracked | Purpose |
|---|---|---|
| `fixtures/hamns-policy-8x4.onnx` | yes | Small interpretable policy used by the tests |
| `*.onnx` (anything else here) | no | Production weights — distribute out of band |

`.gitignore` excludes model files by default and makes an exception for
`models/fixtures/`, so test models travel with the repository and trained weights do not.

## Regenerating the fixture

```bash
python scripts/export_policy_onnx.py
```

Requires `torch` and `onnx`. The script prints the decision the exported policy makes for
a few representative mesh states, which is the quickest way to confirm an export is sane
before wiring it in.

## Using a policy

```bash
cargo run -p hatcher-ux --features onnx
```

```rust
let mesh = hatcher_neural::NeuralMesh::default().try_with_onnx("models/policy.onnx")?;
```
