# Design configuration contract

This is the normative contract for schema version `2.0`. JSON validation uses only the Python standard library. YAML requires PyYAML.

## Top-level shape

Exactly these keys are required:

```text
schema_version, task, observation, model, losses, usage,
baselines, audits, experiments, claims, outputs
```

Unknown top-level keys are rejected. Recipe provenance belongs in `task.provenance`; optional recipe extensions belong in `task.metadata`.

## Task and observation boundary

`schema_version` is the string `"2.0"`.

`task` requires:

- `id`: lowercase stable identifier using letters, digits, hyphens, or underscores;
- `system_id`: the one underlying dynamical/data-generating system;
- `identity_keys`: non-empty unique keys selecting one concrete instance;
- `summary`: observable prediction-task summary;
- optional object-valued `provenance` and `metadata`.

`observation.adapter` requires `kind`, `tokenization`, positive `output_dim`, and two explicit allowlists:

- `context_input_fields`: every entry must occur in at least one declared context source;
- `target_input_fields`: must exactly equal `observation.target.fields`.

`observation.context` requires the task's exact `system_id` and `identity_keys`, an integer window, an empty `target_derived_fields`, and non-empty sources shaped as:

```json
{
  "id": "trajectory",
  "fields": ["observation"],
  "available_through": 0
}
```

Source IDs and fields are non-empty and unique where applicable. Every cutoff is at or before `context.window.end`.

`observation.target` uses the same system and identity keys. Require:

```text
context.start <= context.end < target.start <= target.end
```

Its `fields` are non-empty. Its unique `source_ids` must reference context source IDs, establishing shared dataset/system lineage without exposing target-time values to context.

Each target descriptor is:

```json
{
  "name": "horizon",
  "availability": "known_at_prediction_time",
  "role": "index_only",
  "lineage": {
    "source_id": "prediction-request",
    "source_type": "prediction_request",
    "available_through": 0
  }
}
```

Descriptor source types are `context_source`, `prediction_request`, or `static_metadata`. Roles are `index_only` or `conditioning`.

Each exogenous field has the same shape without `role`. Its source type is `context_source`, `external_plan`, or `static_metadata`. A `context_source` lineage must name a declared source containing that field and cannot exceed that source's cutoff. A prediction request or external plan must be marked `known_at_prediction_time`. All lineages must be available by context end, and conditioning field names cannot reuse predicted target field names.

## OPF state contract

`model` requires an `id`, a shared context/target encoder with `output_dim == d`, and an OPF predictor with `output_dim == d`.

`model.state` requires positive integers `d`, `K`, and `r` with `K * r == d`, plus:

```json
{
  "coordinate_layout": "concatenate",
  "analysis": {
    "kind": "learned_projectors",
    "projector_count": 4,
    "projector_shape": [8, 2],
    "target_stop_gradient": true,
    "orthogonality_mode": "soft_gram"
  },
  "synthesis": {
    "kind": "moore_penrose_pseudoinverse",
    "analysis_map": "projector_transpose",
    "require_full_rank": true
  }
}
```

`analysis.orthogonality_mode` is optional for backward compatibility and
defaults to `soft_gram`. Allowed values are:

- `soft_gram`: optimize the analysis rows directly with the projector-Gram loss;
- `qr_init`: QR-orthonormalize once at initialization, then continue with the
  ordinary learnable rows and Gram loss;
- `qr_retraction`: initialize by QR and re-orthonormalize outside the forward
  path after selected optimizer steps.

`qr_retraction` additionally requires:

```json
{
  "orthogonality_mode": "qr_retraction",
  "qr_retraction": {
    "timing": "after_optimizer_step",
    "frequency": 1,
    "sign_canonicalization": "positive_r_diagonal",
    "training_compute_accounting": "report_separately"
  }
}
```

The retraction frequency is a positive one-based optimizer-step interval. QR
inside the forward path is unsupported. The positive-`R`-diagonal convention
removes arbitrary QR sign flips. All modes retain the projector-Gram contract
and geometry/synthesis audits. Retraction compute is outside predictor-forward
FLOPs and must therefore be reported separately for fair comparisons.

`factors` contains exactly `K` objects and no semantic fields. At position `k`, the only allowed object is `{ "id": "pc_KKK", "index": k, "dim": r }`, with a zero-padded ID such as `pc_003`.

Coordinate concatenation and complete state synthesis are different contracts. Let `P_k` be learned `[d, r]` projectors and let `A = stack(P_k^T)` have shape `[K*r, d] = [d, d]`. Concatenated factor coordinates are `y = A z`. Complete state synthesis is `z_tilde = pinv(A) y`; it is valid only after measuring `rank(A) == d`. The geometry audit also bounds the condition number and reconstruction NMSE.

## Loss contract

All four objects are required and enabled:

1. `factor_prediction`: `kind: "factor_mse"` with positive `weight`.
2. `projector_orthogonality`: `kind: "projector_gram"`, positive `weight`, `components` containing exactly `within_projector` and `cross_projector`, and `projectors` covering all canonical factor IDs.
3. `factor_activity`: `kind: "coordinate_std_floor"`, `scope: "projected_target_factors"`, positive `weight`, `min_std`, and `epsilon`; `coordinates` must enumerate every `{factor_id, coordinate_index}` in canonical factor-major order.
4. `online_encoder_activity`: `kind: "coordinate_std_floor"`, `scope: "online_context_encoder"`, positive `weight`, `min_std`, and `epsilon`; `dimensions` must be exactly `0..d-1`.

These fields preserve the intended semantics: factor prediction compares predicted blocks with stop-gradient projected targets; projector-Gram pressure contains both within-projector orthonormality and cross-projector orthogonality; the two standard-deviation floors separately protect projected target factors and online encoder coordinates. Configuration does not prove any measured floor is met.

## Capacity contract and baselines

`model.capacity` requires:

```json
{
  "full_model_trainable_parameters": 12000,
  "predictor_flops": 24000,
  "training_steps": 1000,
  "parameter_tolerance": 0.003,
  "predictor_flops_tolerance": 0.02,
  "parameter_scope": "full_trainable_model",
  "flops_scope": "predictor_forward_per_target"
}
```

Require exactly one `standard_jepa` and exactly one `unconstrained_multihead`. Both have `predictor.output_dim == d`. The multihead baseline additionally has `heads == K`, `head_dim == r`, and `constraints: []`.

Every required baseline's `capacity_match` names the reference model and explicitly repeats:

- positive `full_model_trainable_parameters` within `parameter_tolerance`;
- positive `predictor_flops` within `predictor_flops_tolerance`;
- exactly matching `training_steps`;
- both tolerance values and both counting scopes;
- `encoder_policy: "same_family_and_width"`;
- `data_policy: "same_samples_and_augmentations"`.

Declared budgets are plans. Generated capacity stubs require fresh measured counts for the reference and both baselines before an audit can pass. The generated activity stub likewise requires measured standard deviations with exact `[K][r]` projected-factor and `[d]` online-encoder coverage; configured floors are not treated as measurements.

## Use modes

Choose exactly one mode and provide only its detail object:

- `terminal_readout`: `readout.tasks` and `readout.metrics` are non-empty string lists;
- `repeated_transition`: `transition.rollout_horizons` is a unique ascending list of positive integers; `condition_on_exogenous` is boolean;
- `factor_analysis`: non-empty `analysis.probes` and `analysis.interventions`, plus `coordinate_reporting: "prediction_coordinates_only"`.

## Audits

`audits.geometry` is enabled and includes all of:

```text
state_shape, coordinate_concatenation,
within_projector_orthonormality, cross_projector_orthogonality,
analysis_full_rank, analysis_condition_number,
pseudoinverse_state_synthesis
```

It also requires `full_rank_required: true`, positive `max_condition_number`, and non-negative `max_synthesis_nmse`.

`audits.factor_activity` uses `metric: "coordinate_std"`, a positive `minimum`, and the complete factor-dimension coordinate list. `audits.online_encoder_activity` uses the same metric/minimum and exact dimensions `0..d-1`.

`audits.target_leakage` includes `system_identity`, `temporal_separation`, `feature_lineage`, `descriptor_availability`, `adapter_lineage`, and `conditioning_lineage`.

`audits.capacity_matching` is enabled, has `measured_counts_required: true`, and covers both required baseline IDs. `audits.claim_coverage` is enabled.

## Structured experiment and claim evidence

Every experiment contains exactly interpretable evidence metadata:

```json
{
  "id": "rollout-comparison",
  "system_id": "synthetic-system-v1",
  "usage_mode": "repeated_transition",
  "split": "test",
  "horizons": [1, 4],
  "metrics": [
    {"id": "normalized_rollout_mse", "direction": "lower_is_better"}
  ],
  "baselines": ["standard_jepa", "unconstrained_multihead"],
  "audits": [
    "geometry", "factor_activity", "online_encoder_activity",
    "target_leakage", "capacity_matching"
  ],
  "tests_claims": ["planned-rollout-comparison"]
}
```

Experiment objects contain only `id`, `system_id`, `usage_mode`, `split`, `horizons`, `metrics`, `baselines`, `audits`, and `tests_claims`; result values and conclusion payloads are forbidden at design time. Horizons must exactly equal selected repeated-transition horizons; they are `[]` for the other modes. Metric directions are `lower_is_better` or `higher_is_better`.

Claims contain exactly these keys and no prose:

```text
id, status, model_id, metric_id, direction, split, horizons,
baseline_ids, audit_ids, experiment_ids, usage_mode
```

`status` is always `planned`. Model, metric/direction, split, horizons, both required baseline kinds, use mode, and reciprocal experiment links must match every referenced experiment in both directions. Every claim requires geometry, projected-factor activity, online-encoder activity, target-leakage, and capacity-matching audits. Consequently, changing a four-step MSE claim to “accuracy at 100 steps” without changing actual experiment coverage deterministically fails.

Passing claim coverage establishes only that the declared experiment could test the structured claim. It never changes status to `supported` and never proves a conclusion.

## Output boundary

`outputs.config_formats` is a non-empty subset of `json` and `yaml`. The scaffold uses Python and `include_training_loop` must be `false`.

## Stable diagnostic codes

Downstream automation should branch on codes rather than messages.

| Area | Success | Representative failures |
|---|---|---|
| schema | `SCHEMA_VALID` | `SCHEMA_VERSION_UNSUPPORTED`, `SCHEMA_REQUIRED`, `SCHEMA_UNKNOWN_KEY` |
| system | `SYSTEM_IDENTITY_VALID` | `SYSTEM_IDENTITY_MISMATCH` |
| leakage | `TARGET_LEAKAGE_CONTROLS_VALID` | `LEAKAGE_TEMPORAL_OVERLAP`, `LEAKAGE_TARGET_DERIVED_CONTEXT`, `LEAKAGE_SOURCE_UNAVAILABLE`, `LEAKAGE_SOURCE_UNKNOWN`, `LEAKAGE_INPUT_UNAVAILABLE` |
| state | `STATE_COMPOSITION_VALID` | `STATE_DIMENSION_MISMATCH`, `STATE_COMPOSITION_INVALID`, `COORDINATE_CONTRACT_INVALID`, `ORTHOGONALITY_STRATEGY_INVALID` |
| activity | `FACTOR_ACTIVITY_CONFIGURED` | `FACTOR_ACTIVITY_INCOMPLETE` |
| baselines | `BASELINES_CAPACITY_MATCHED` | `BASELINE_REQUIRED_MISSING`, `BASELINE_CAPACITY_MISMATCH`, `BASELINE_FLOPS_MISMATCH`, `BASELINE_UNCONSTRAINED_INVALID` |
| use mode | `USAGE_MODE_VALID` | `USAGE_MODE_INVALID`, `USAGE_MODE_CONTRACT_INVALID` |
| claims | `CLAIM_COVERAGE_COMPLETE` | `CLAIM_EXPERIMENT_MISSING`, `CLAIM_RECIPROCITY_MISSING`, `CLAIM_METRIC_UNCOVERED`, `CLAIM_DIRECTION_MISMATCH`, `CLAIM_SPLIT_MISMATCH`, `CLAIM_HORIZON_MISMATCH`, `CLAIM_BASELINE_UNCOVERED`, `CLAIM_AUDIT_UNCOVERED` |
| output | `OUTPUT_BOUNDARY_VALID` | `OUTPUT_TRAINING_FORBIDDEN` |

Malformed fields can also produce `FIELD_INVALID`. The validator CLI returns `0` for valid design, `1` for design errors, and `2` for parse/load/I/O failures.
