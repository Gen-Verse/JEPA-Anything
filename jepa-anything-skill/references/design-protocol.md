# Task-design protocol

Use this protocol to translate task intent into a proposal that deterministic code can inspect. It designs interfaces and evidence; it does not run training.

## 1. Fix one prediction boundary

Write the task as:

```text
Given declared fields from one SYSTEM instance, identified by IDENTITY_KEYS,
through CONTEXT_END, predict TARGET_FIELDS after TARGET_START, conditioned only
on descriptors and exogenous inputs auditable as available at prediction time,
so that exactly one DOWNSTREAM_USE consumes the resulting state.
```

If two sources are sensors of one system, document that shared system and instance-key join. If they are different systems, define an explicit joint system or reject the pairing. A common dataset label is not a shared instance.

Record two adapter allowlists. Every context input must resolve to a declared context source and cutoff. Target inputs must exactly match target fields. Trace each descriptor and exogenous input to a source ID, source type, and availability cutoff. Do not relabel an unknown future action/observation as an external plan.

## 2. Choose one consumer

- `terminal_readout`: state is consumed once. Declare readout tasks and metrics; do not imply recurrent dynamics.
- `repeated_transition`: state is repeatedly advanced. Declare unique ascending rollout horizons and whether every step requires known exogenous inputs.
- `factor_analysis`: coordinates are probed or intervened upon. Report only canonical prediction-coordinate IDs.

The architecture does not choose the mode. The downstream consumer does.

## 3. Separate coordinate analysis from state synthesis

Choose positive `d`, `K`, and `r` with `K*r = d`. Use the smallest auditable decomposition compatible with the downstream capacity envelope.

For learned projectors `P_k` of shape `[d, r]`, define the analysis map

```text
A = stack(P_0^T, ..., P_{K-1}^T)  with shape [K*r, d] = [d, d].
```

The factor-coordinate interface is the canonical concatenation

```text
y = concat(P_0^T z, ..., P_{K-1}^T z) = A z.
```

This concatenation is not itself a reconstruction of `z`. Complete state synthesis is

```text
z_synth = pinv(A) y.
```

Therefore require learned projectors, a Moore-Penrose pseudoinverse synthesis interface, measured `rank(A) == d`, a condition-number ceiling, and a synthesis-NMSE ceiling. Exact orthonormal projectors make `A^T` a special-case inverse; the general interface remains `pinv(A)` so imperfect learned geometry is auditable.

Coordinate IDs are positional (`pc_000`, `pc_001`, …), not semantic. A domain interpretation is a future probe/intervention hypothesis, never a generated factor name.

Choose one orthogonality strategy. `soft_gram` is the default and learns the
analysis rows directly. `qr_init` performs a single sign-canonicalized QR before
learning. `qr_retraction` performs the same initialization and exposes an
explicit hook after optimizer steps at a declared positive frequency. Do not
apply QR inside the forward path. Retraction changes neither the analysis
interface nor the general pseudoinverse synthesis contract, and it does not
waive Gram, rank, conditioning, synthesis-NMSE, or compute audits.

## 4. Preserve four distinct loss semantics

Configure these separately:

1. Factor prediction MSE compares each predicted factor block with its stop-gradient projected target block:

   ```text
   L_factor = mean_k ||factor_hat_k - stopgrad(P_k^T z_target)||^2.
   ```

2. Projector-Gram loss contains both within-projector and cross-projector terms:

   ```text
   L_gram = sum_k ||P_k^T P_k - I||_F^2
          + sum_{k != l} ||P_k^T P_l||_F^2.
   ```

3. Factor activity applies a standard-deviation floor to every scalar coordinate of every projected target factor, with explicit `weight`, `min_std`, and numerical `epsilon`.

4. Online encoder activity independently applies a standard-deviation floor to every dimension `0..d-1` of the online context encoder, with its own `weight`, `min_std`, and `epsilon`.

Do not merge the two activity losses: one monitors projected target factors, the other protects the online encoder. Do not call a Gram penalty a data cross-covariance loss.

## 5. Match baseline capacity on three axes

Always include one standard JEPA and one unconstrained `K`-head predictor of width `r`. Use the same encoder family/width and the same data samples/augmentations.

Declare, and later measure, all three axes:

- full-model trainable parameters under `full_trainable_model` scope;
- predictor-forward FLOPs per target under `predictor_forward_per_target` scope;
- identical training steps.

Parameter and predictor-FLOPs tolerances are separate. A parameter-matched predictor can still have materially different compute; a FLOPs-matched predictor can still hide capacity elsewhere in the model. The generated capacity stub accepts measured records, not a conclusion copied from configuration.

## 6. Design audits before claims

Cover at least:

- system/instance identity and temporal/feature/adapter/conditioning lineage;
- state and factor shapes plus canonical concatenation;
- within-projector orthonormality and cross-projector orthogonality;
- analysis-map rank and condition number;
- pseudoinverse synthesis NMSE;
- per-factor-dimension and per-online-encoder-dimension standard deviation;
- measured parameter/FLOPs/step matching;
- use-mode fitness against both baselines.

A rollout metric must cover every declared horizon. Terminal and factor-analysis experiments use an empty horizon list. Factor-analysis conclusions need the configured probes/interventions; prediction loss alone cannot establish structure.

## 7. Compile claims into exact evidence keys

Do not place free-form claim prose inside `claims`. Represent each planned comparison with exact fields:

```text
model, metric, direction, split, horizons, baseline IDs, audit IDs,
experiment IDs, and usage mode.
```

Every referenced experiment reciprocally names the claim and matches all fields in both directions. Both baseline families and all substantive geometry, activity, leakage, and capacity audits are mandatory. This exact join is what prevents a changed “100-step accuracy” assertion from passing against an unchanged four-step MSE experiment.

`status: planned` means testable by the declared experiment. It never means supported. Result values, uncertainty, and statistical decisions belong to a separate execution workflow.

## 8. Deliver an inspectable handoff

Deliver:

1. validated JSON or YAML;
2. deterministic validation report;
3. generated adapter, OPF state, geometry/activity, use-mode, capacity, and evidence-audit interfaces;
4. unresolved data-owner assumptions.

Run the generated contract tests. Do not deliver weights, checkpoints, an optimizer, a training loop, or empirical claims from this skill.
