# Synthetic linear dynamics

This is a minimal, deterministic audit fixture for a JEPA Anything task. It
uses a stable four-dimensional state with a known scalar control input and a
six-dimensional observation projection. The fixture is deliberately small:
its purpose is to reveal task-design errors before any costly training is
attempted.

## Files

- `recipe.json` is the source task specification.
- `design.expected.json` is the expected task-designer output and is intended
  to pass `jepa-anything-skill/scripts/validate_design.py`.
- `run_recipe.py` generates trajectories in memory and audits structural and
  activity invariants using only the Python standard library.

## Run

From the repository root:

```bash
python3 recipes/synthetic-linear-dynamics/run_recipe.py \
  --output work/synthetic-linear-dynamics-report.json
```

For a quick check that leaves no file behind:

```bash
python3 recipes/synthetic-linear-dynamics/run_recipe.py --quiet
```

The run is deterministic for the recorded Python algorithm, seed, matrices,
and sample count. Floating-point summaries can still differ at the last bits
across Python/platform implementations; pass/fail thresholds include ample
margin and are not intended as benchmark tolerances.

## What is and is not tested

The script checks:

1. context and target selectors use the same underlying system and trajectory
   identity;
2. context/target adapter fields, descriptors, and exogenous inputs have
   prediction-time source lineage;
3. context offsets do not intersect or follow target offsets;
4. descriptors and preprocessing do not read target observations;
5. `K*r=d`, factor coordinates are concatenated only as an analysis layout,
   and state synthesis is declared through the Moore-Penrose pseudoinverse;
6. the loss plan uses factor MSE, within/cross projector-Gram geometry, and
   separate coordinate-wise activity floors for targets and online encoding;
7. every coordinate of both synthetic oracle groups varies in generated
   training trajectories;
8. standard-JEPA and unconstrained-multihead baselines declare full-model
   trainable-parameter and predictor-FLOP matching plans;
9. the repeated-transition usage mode has a multi-step rollout experiment;
10. every structured claim has direction-, split-, horizon-, baseline-, and
    audit-matched experimental coverage.

Synthetic latent coordinates are used only as an audit oracle. The two groups
remain named `pc_000` and `pc_001`; no physical, medical, causal, or semantic
meaning is assigned to them. The recipe is a Skill example and makes no
performance or factor-interpretability claim.
