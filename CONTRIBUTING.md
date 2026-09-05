# Contributing

Thank you for helping improve JEPA Anything.  Contributions should preserve the
separation between proposal, validation, execution, and evidence.

## Development checks

```bash
python -m pip install -e './jepa-anything-core[dev]'
make check
```

Add tests for observable invariants rather than generated prose.  In
particular, validator changes should include both an accepted design and a
minimal failing design with a stable diagnostic code.

## Design rules

1. Keep `jepa-anything-core` independent of any LLM provider.
2. Keep deterministic policy in validator code and schemas, not only in a
   prompt.
3. Never infer semantics from latent position, variance, correlation, or a
   convenient label.  Use neutral coordinate identifiers such as
   `pc_003`.
4. Keep factor-coordinate concatenation distinct from state synthesis. A
   repeated-transition design must state how the full latent state is recovered
   from the analysis coordinates.
5. Activity plans are coordinate-wise for projected targets and online encoder
   representations; a factor-level mean must not mask an inactive coordinate.
6. Capacity comparisons must state the full-model trainable-parameter scope,
   predictor-FLOP measurement, and tolerances for both.
7. A recipe distinguishes expected criteria from observed results and records
   the command, seed, data fingerprint, code revision, and artifact digest.
8. Claims are structured links among usage mode, experiments, metrics,
   baselines, and audits. Free-form prose is not evidence coverage.
9. Do not commit large model weights.  Commit a manifest with a content digest
   and a retrieval location when redistribution is permitted.

## Pull requests

Keep changes scoped to one contract or capability where possible.  Describe:

- which invariant or user workflow changes;
- how the change is tested;
- whether old design configurations remain valid;
- what new scientific or operational claim, if any, becomes possible.

Do not include secrets, private datasets, access tokens, or unverifiable result
claims.
