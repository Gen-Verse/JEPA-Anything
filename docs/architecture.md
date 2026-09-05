# Architecture and trust boundaries

## Components

### World-model compiler Skill

`jepa-anything-skill` is implicitly discoverable for world-model requests. It
translates the user's ordinary domain task—or facts recovered from an existing
repository—into a declarative JEPA Anything design without requiring the user
to supply JEPA terminology. The proposal contains observation/token adapters, context and target
views, target descriptors, exogenous inputs and their input lineage, an encoder
recommendation, `d/K/r`, factor-coordinate packing, state synthesis, usage
modes, metrics, capacity plans, baselines, experiments, and structured claims.

Missing causal facts that could change the task or introduce leakage are
blocking; missing JEPA architecture choices are not. The compiler asks only for
the former and records its proposed adapter, encoder, dimensions, and capacity
choices as inspectable assumptions.

System-level equality is not enough for a positive pair: every design also
declares non-empty instance identity keys (for example an episode, subject, or
trajectory key), and context/target must carry the same key tuple.

The LLM may fill this contract but cannot waive validator diagnostics.  A valid
contract can be rendered into an implementation skeleton.  The renderer has no
training loop side effect, network side effect, or checkpoint side effect.

### Core support library

`jepa-anything-core` supplies tensor-level, LLM-independent mechanisms used by
generated implementations: OPF projection and composition, regularizers,
matched baselines, and numerical geometry audits. Its public APIs accept
explicit dimensions and tensors; they do not interpret a latent dimension as a
physical or clinical concept.

### Recipes and checkpoints

A recipe pins an example design, deterministic seeds, expected checks, and
commands so the Skill can be exercised end to end. An expected check is not an
observed result. Observations belong in run manifests produced by executing a
recipe. A checkpoint manifest pins the exact recipe and compiled design,
records execution provenance and evidence boundaries, and lists size/hash
metadata for any future artifact.

## Trust boundary

| Stage | May be generative? | Deterministic gate | Output |
|---|---:|---:|---|
| World-model intent and repository interpretation | yes | causal-boundary review + schema validation | normalized task |
| JEPA Anything compilation | yes | schema validation | design proposal |
| Adapter/context/target proposal | yes | system + leakage checks | checked views |
| OPF shape proposal | yes | `K*r=d` + coordinate packing + synthesis checks | checked geometry |
| Baseline proposal | yes | presence + parameter/FLOP capacity checks | comparison plan |
| Claim proposal | yes | experiment/metric coverage | claim matrix |
| Skeleton rendering | no | validated input required | source scaffold |
| Training | outside Skill | recipe/run controls | measured artifacts |
| Semantic interpretation | human/scientific workflow | intervention evidence | bounded claim |

## Failure semantics

Validator diagnostics use stable machine-readable codes and a severity.  An
error prevents skeleton generation.  A warning identifies a design risk that
does not contradict the contract.  A passing validation says nothing about
empirical performance.

## State-interface boundary

The task designer compiles a problem into an implementation contract. In
particular, concatenating the `K` predicted `r`-dimensional blocks creates the
analysis-coordinate vector `u`, while a separate Moore-Penrose pseudoinverse
synthesis maps `u` through the transpose of the learned projector analysis map
back to a complete `d`-dimensional latent state. A design must not call
concatenation alone state synthesis.

Likewise, an activity requirement is coordinate-wise: one high-variance
coordinate must not hide a collapsed coordinate in the same factor. A capacity
comparison must plan measurements for full-model trainable parameters and
predictor FLOPs, not merely repeat the same nominal hidden width. These are
deterministic design checks; only executed, matched experiments can support
performance or scientific claims.

## Coordinate naming policy

Names such as `disease_factor`, `speed_latent`, or `intent_axis` imply a result
that has not yet been established.  Design-time identifiers use neutral names,
for example `pc_000`.  A separate, evidence-linked analysis
may propose an interpretation after the required experiments are complete.
