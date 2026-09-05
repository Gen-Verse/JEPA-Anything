# JEPA Anything Core

`jepa-anything-core` is a small, typed PyTorch library for building and auditing
orthogonal predictive-factor (OPF) state interfaces. It contains reusable model
primitives—not a task generator, dataset pipeline, or claim of experimental
performance.

The central contract is:

```text
d = K × r
state (..., d) ⇄ predictive coordinates (..., K, r)
```

Here `K` is the number of factor groups and `r` is the width of each group. A
complete orthonormal basis makes decomposition and synthesis lossless up to
floating-point error. Factor indices are deliberately anonymous. The library
never calls a coordinate a “disease”, “speed”, or other semantic factor without
separate experimental evidence.

## Included

- Fixed orthonormal bases and directly learned analysis projectors. Learnable
  projectors are optimized without a forward-time QR rewrite.
- State analysis, general Moore–Penrose synthesis, explicit transpose synthesis
  for audits, original-space factor components, and rank-`r` projectors.
- Projector-Gram orthogonality, per-coordinate factor activity, and
  token-mask-aware online-encoder variance losses.
- Pure factor regression and composite objective construction;
  optimizer and EMA scheduling remain explicit responsibilities of generated
  downstream execution code.
- Statistical factor decorrelation and checkpointable Welford variance tracking
  as optional diagnostics, kept separate from online-encoder activity.
- A standard EMA-target JEPA baseline and an unconstrained multi-head ablation.
- Exact predictor capacity matching: a `d`-wide output and `K` independent
  `r`-wide outputs contain the same number of parameters when `d = Kr`.
- JSON-serializable basis, activity/correlation, and reconstruction audits.

Geometric orthogonality and statistical decorrelation are kept distinct.
`projector_orthogonality_loss` applies the within- and cross-projector Gram
objective directly to the learned analysis rows. `factor_decorrelation_loss`
examines sample statistics only and is not a substitute for geometric
orthogonality. Neither establishes semantic independence.

Factor activity and encoder variance use a linear standard-deviation hinge for
every coordinate. Encoder variance operates on the current online context
representation and accepts a `valid_mask` for padded tokens. All variance
statistics use population variance (`correction=0`); float16 and bfloat16
reductions are accumulated in float32. The Welford tracker is retained for
streaming monitoring, not as the online-encoder objective.

## Install

```bash
python -m pip install -e ./jepa-anything-core
```

Only PyTorch is required at runtime. For local tests:

```bash
python -m pip install -e './jepa-anything-core[dev]'
python -m pytest jepa-anything-core/tests -q
```

## Minimal OPF usage

```python
import torch
from jepa_anything_core import (
    OrthogonalFactorProjection,
    audit_opf_geometry,
    factor_activity_loss,
    projector_orthogonality_loss,
)

opf = OrthogonalFactorProjection(
    state_dim=12,
    num_factors=3,
    factor_dim=4,
    learnable=True,
)

state = torch.randn(128, 12)
coordinates = opf(state)              # (128, 3, 4)
reconstructed = opf.compose(coordinates)

loss = (
    projector_orthogonality_loss(opf.analysis_basis())
    + factor_activity_loss(coordinates, min_std=0.1)
)
report = audit_opf_geometry(opf, state)
assert report.basis.passed and report.round_trip.passed
```

`report.factors.passed` depends on the configured activity and correlation
thresholds and on the observed samples; it should be logged rather than assumed.
When an audit tolerance is omitted, defaults are dtype-aware: `1e-10` for
float64, `1e-5` for float32, `1e-2` for float16, and `5e-2` for bfloat16. The
effective value is stored in the report.

Operational synthesis always uses the Moore–Penrose inverse of the current
analysis map. `opf.transpose_synthesize(coordinates)` is intentionally separate:
its error reveals whether the learned map has reached self-dual orthogonal
geometry.

## Online-encoder variance and streaming monitoring

```python
from jepa_anything_core import encoder_variance_loss

# context_tokens: (batch, tokens, d); valid_tokens: (batch, tokens)
loss = encoder_variance_loss(context_tokens, valid_mask=valid_tokens, min_std=0.1)
```

For a diagnostic that aggregates very small batches over time:

```python
from jepa_anything_core import OnlineVarianceLoss

variance_monitor = OnlineVarianceLoss(num_factors=3, factor_dim=4, min_std=0.1)
monitor_value = variance_monitor(coordinates)  # updates in train mode

checkpoint = variance_monitor.state_dict()  # count, mean, and M2 are preserved
```

The monitor uses a differentiable Welford merge: history is detached, while the
current batch receives gradients. In evaluation mode its state is not updated
unless `update=True` is supplied explicitly. It does not replace the
current-batch `encoder_variance_loss` term in the composite objective.

An exactly constant factor is a symmetric stationary point of both variance
objectives: its loss is positive but its gradient is zero. The objectives can
oppose near-collapse, but cannot independently break perfect collapse. Use a
non-degenerate initialization and an asymmetric predictive learning signal.

## Capacity-matched baselines

```python
from torch import nn
from jepa_anything_core import (
    StandardJEPABaseline,
    UnconstrainedMultiHeadJEPABaseline,
    build_capacity_matched_predictors,
    capacity_match_report,
)

standard_predictor, multihead_predictor = build_capacity_matched_predictors(
    input_dim=256,
    state_dim=128,
    num_factors=8,
    factor_dim=16,
    hidden_dim=512,
    depth=3,
)
assert capacity_match_report(standard_predictor, multihead_predictor).matched

# Full experiment builders may also supply measured prediction FLOPs.
assert capacity_match_report(
    standard_predictor,
    multihead_predictor,
    reference_flops=1_000_000,
    candidate_flops=1_010_000,
    flop_tolerance=0.02,
).matched

online_encoder = nn.Linear(1024, 128)
standard = StandardJEPABaseline(online_encoder, standard_predictor)

multihead = UnconstrainedMultiHeadJEPABaseline(
    nn.Linear(1024, 128),
    multihead_predictor,
)
```

Both baselines return a flat prediction and a stop-gradient target. The
multi-head output also exposes its `(..., K, r)` head view. After each optimizer
step, call `model.update_target_encoder()` to perform the EMA update. Capacity
matching covers **only the predictor pair returned by the builder**. It is not a
claim that arbitrary full models are matched. Experiment code must separately
report full-model trainable parameter counts and keep encoders, adapters,
conditioning inputs, optimization, and evaluation protocols identical.

## Audit boundaries

The geometry audits answer only questions that tensors can settle:

- Does `Kr = d` and is the basis full-rank and orthonormal?
- What are its minimum singular value, condition number, and cross-subspace
  overlap?
- Do factor projectors sum to the identity?
- Does pseudoinverse composition reconstruct the state, and what is the direct
  transpose-synthesis NMSE?
- Is every coordinate of every factor active on the audited sample?
- How correlated are distinct factor coordinates?

They do **not** detect observation/target leakage, decide whether context and
target come from one underlying system, assign semantics to coordinates, or
prove that an experiment supports a scientific conclusion. Those checks belong
to the deterministic task validator and experiment manifest built above this
core package.
