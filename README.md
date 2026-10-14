# Gauge Attention

Gauge Attention is a compact PyTorch reference implementation of
**gauge-inspired learned rotary transport** for self-attention. It asks one
focused research question:

> What changes when the rotary phase increment is learned on every sequence
> edge, instead of being fixed across all positions?

The repository turns that question into an inspectable attention layer with
exact mathematical checks. It is intended for research, teaching, and
experimentation—not as evidence that the mechanism improves model quality.

## Why this project exists

Rotary Position Embeddings (RoPE) rotate query and key coordinates using a
fixed phase increment. That makes the phase between two tokens depend only on
their relative distance. Gauge Attention keeps the same efficient 2D rotations
but promotes the fixed increment to a learned, position-dependent parameter.

This small change creates a useful continuum between familiar cases:

| Connection | Result |
| --- | --- |
| Every edge is zero | Position-free attention |
| Every edge uses the standard RoPE frequencies | Standard RoPE |
| Edge values vary with position | Learned position-dependent transport |

The implementation includes all three variants behind matching attention
interfaces so their logits and outputs can be compared directly.

## Concept in one minute

For attention head `h`, coordinate pair `p`, and sequence edge `k`, let
`A[h, p, k]` be the angle added while moving from position `k` to `k + 1`.
Accumulating these edge values defines a local frame at every position:

```text
theta[h, p, 0] = 0
theta[h, p, i] = sum(A[h, p, k], k = 0 ... i - 1)
```

The transport from query position `i` to key position `j` is the phase
difference

```text
Phi[h, p, i, j] = theta[h, p, j] - theta[h, p, i].
```

It is therefore antisymmetric and compositional:

```text
Phi(i, j) = -Phi(j, i)
Phi(i, j) + Phi(j, k) = Phi(i, k).
```

Queries and keys are rotated once by their absolute phases before ordinary
scaled dot-product attention. The forward path does not construct the full
pairwise transport tensor. When all edges share the standard RoPE increment,
`theta[i] = i * omega`, recovering RoPE exactly.

For a longer derivation and a precise discussion of scope, see
[the concept note](docs/concept.md).

## Design goals and implemented properties

- **Small and inspectable:** the mechanism is separated into tensor functions
  and explicit attention layers rather than hidden behind a training stack.
- **Differentiable:** the connection is a regular `torch.nn.Parameter`; tests
  verify that gradients reach both the input and learned edge values.
- **Reference-compatible:** constant RoPE and zero connections are checked
  against independent `RoPEAttention` and `NoPositionAttention` references.
- **No stale phase cache:** phases are accumulated from the current connection
  on every call, so optimizer updates are immediately reflected.
- **Explicit sequence boundaries:** extending beyond the learned connection is
  an opt-in modeling decision rather than silent behavior.
- **PyTorch-native state:** parameters participate normally in device moves,
  optimizers, and `state_dict` save/load operations.
- **Optional value transport:** queries and keys are rotated by default;
  `rotate_values=True` additionally transports values and rotates the result
  back into each query frame.

## Installation

Gauge Attention requires Python 3.10 or newer and PyTorch 2.1 or newer.

```bash
git clone https://github.com/topppg/gauge-attention.git
cd gauge-attention
python -m pip install -e .
```

For development, install the test and lint tools:

```bash
python -m pip install -e '.[dev]'
```

## Quick start

```python
import torch
from gauge_attention import GaugeAttention

layer = GaugeAttention(
    embed_dim=256,
    n_heads=8,
    max_positions=512,
    connection_init="rope",  # "rope" or "zeros"
    boundary="error",
)

x = torch.randn(2, 128, 256)
y = layer(x, is_causal=True)

assert y.shape == x.shape
assert layer.connection.shape == (8, 16, 511)
```

`GaugeAttention` expects tensors shaped `(batch, positions, embed_dim)`.
`embed_dim` must be divisible by `n_heads`, and each attention head must have
an even dimension because the final axis is divided into 2D rotation planes.

### Inspect the learned geometry

The layer exposes both accumulated and pairwise phases for analysis:

```python
absolute = layer.absolute_phases(seq_len=128)
transport = layer.transport_phases(seq_len=128)

assert absolute.shape == (8, 128, 16)
assert transport.shape == (8, 128, 128, 16)
```

`transport_phases` materializes all query/key pairs and is intended for
inspection. Normal forward passes use only the linear-size absolute phases.

Run the complete small example with:

```bash
python examples/inspect_transport.py
```

## Main API

| Symbol | Purpose |
| --- | --- |
| `GaugeAttention` | Self-attention with learned rotary edge increments |
| `RoPEAttention` | Minimal fixed-frequency RoPE reference |
| `NoPositionAttention` | Position-free attention reference |
| `connection_to_absolute_phases` | Accumulate edge increments into position phases |
| `pairwise_transport_phases` | Compute `theta[key] - theta[query]` for inspection |
| `rope_connection` | Construct the constant connection equivalent to RoPE |
| `rope_frequencies` | Construct standard RoPE angular increments |
| `rotate_pairs` | Apply one 2D rotation per coordinate pair |

Both boolean masks (`True` means allowed) and floating-point additive masks are
supported in 2D, batched 3D, or batch/head 4D forms. Causal masking is enabled
with `is_causal=True`.

## Behavior beyond `max_positions`

The learned connection contains `max_positions - 1` edges. Longer inputs fail
by default instead of silently inventing new parameters:

```python
GaugeAttention(..., boundary="error")  # default: raise
GaugeAttention(..., boundary="repeat_last")  # repeat the final edge increment
GaugeAttention(..., boundary="zero")  # append zero increments
```

These are precise computational definitions, not claims that a trained model
will generalize to longer sequences.

## Verified results

The following results describe engineering correctness checks in this
repository. They are **not** training or model-quality benchmarks.

| Check | Verified outcome |
| --- | --- |
| Unit test suite | 24 tests passed; 1 CUDA-only test skipped when CUDA was unavailable |
| Zero-connection equivalence | Logits and outputs match position-free attention |
| Constant-connection equivalence | Logits and outputs match the included RoPE reference |
| Transport identities | Antisymmetry and interval composition hold in tests |
| Differentiation | Gradients reach inputs and learned connection parameters |
| Parameter updates | Updated connections are used immediately; no stale cache |
| Tensor behavior | Shapes, float32/float64 dtypes, masks, CPU, optional CUDA, and serialization are covered |
| Boundary behavior | `error`, `repeat_last`, and `zero` are tested at and beyond the learned range |
| Package quality | Ruff lint and format checks pass; source and wheel distributions build successfully |

Reproduce the checks locally:

```bash
ruff check .
ruff format --check .
pytest -q
uv build
```

GitHub Actions runs linting and tests on Python 3.10 and 3.12 for every push
and pull request.

## What this repository does not claim

This is an alpha-stage research prototype. It currently provides:

- a mechanism and reference implementation;
- exact equivalence and mathematical property checks;
- a foundation for controlled experiments.

It does **not** currently provide:

- a paper, pretrained weights, or a training recipe;
- language-model or downstream-task benchmarks;
- evidence of better accuracy, efficiency, extrapolation, or convergence;
- production-optimized attention kernels;
- a general non-Abelian gauge construction.

The current implementation uses independent real-valued rotations in 2D
coordinate planes and explicit attention code for clarity. API changes may
occur before version 1.0.

## Project structure

```text
gauge-attention/
├── src/gauge_attention/
│   ├── attention.py       # Gauge, RoPE, and position-free attention layers
│   └── functional.py      # Transport, phase, frequency, and rotation helpers
├── tests/                 # Equivalence, gradient, mask, boundary, and state tests
├── examples/
│   └── inspect_transport.py
├── docs/
│   └── concept.md         # Mathematical motivation and scope
└── .github/workflows/
    └── tests.yml          # Python 3.10/3.12 continuous integration
```

## Roadmap

Useful next steps are deliberately experimental:

- compare initialization and regularization choices for learned connections;
- evaluate behavior across sequence lengths and controlled synthetic tasks;
- separate content-conditioned from purely position-indexed transport;
- study parameter sharing across heads, layers, and coordinate pairs;
- prototype optimized kernels only after establishing useful behavior;
- extend the mathematical treatment beyond independent 2D rotations.

Roadmap items are directions for investigation, not promised results.

## Contributing

Issues and focused pull requests are welcome, especially for mathematical
checks, clearer examples, numerical edge cases, and reproducible experiments.
Please keep empirical claims tied to runnable code and reported methodology.

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
pytest
```

## Citation

There is no associated paper or archival citation yet. If this implementation
supports your work, link to the repository and record the exact version or
commit used so the experiment remains reproducible.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
