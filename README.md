# JEPA Anything

[简体中文](README.zh-CN.md)

JEPA Anything is a Codex Skill that automatically converts a natural-language
world-model task into a **verifiable JEPA Anything world-state interface**. The Skill is the product;
the rest of the repository provides its deterministic validators, generated
project skeleton, reusable tensor checks, and maintained examples:

```text
jepa-anything-skill/   Skill instructions, validator, schema, and scaffold assets
jepa-anything-core/    deterministic OPF/loss/baseline/audit support library
recipes/               example task specifications used to exercise the Skill
checkpoints/           optional artifact metadata contract
```

The central rule is simple:

> The LLM proposes a design; deterministic code decides whether the design is
> internally valid.

The Skill is therefore **not** an autonomous trainer and does not claim to
discover named semantic factors.  It turns a user's observation process,
prediction task, and downstream use into an inspectable contract.  Latent
components remain neutral *predictive coordinates* until interventions or
other experiments establish a semantics.

## What is verified

Every generated design must pass machine-checkable gates for:

- shared system and concrete-instance identity between context and target;
- absence of target leakage through observations, descriptors, and exogenous
  inputs;
- the OPF dimension invariant `K * r == d`, factor-coordinate packing, and an
  explicit analysis-to-state synthesis rule;
- a per-coordinate activity plan for projected targets and the online encoder;
- standard-JEPA and unconstrained-multi-head baselines with explicit total
  trainable-parameter and predictor-FLOP matching plans;
- exactly one declared use mode: terminal readout, repeated transition, or
  factor analysis;
- experiment coverage for every claim the design proposes to evaluate.

Passing these checks means only that the **design contract is coherent**.  It
does not establish factor semantics, causal identification, scientific truth,
or model quality.

## Requirements

- Python 3.10 or newer.
- Codex for natural-language Skill invocation.
- PyYAML only when reading or writing YAML; JSON needs no third-party package.
- PyTorch 2.1 or newer only when using `jepa-anything-core` tensor operations.

The validator, scaffold generator, and generated contract tests use the Python
standard library when the design is JSON.

## Supported world-model tasks

The Skill applies when the task predicts the future of the same underlying
system from current or historical information:

| Source task | Typical inputs | Primary converted mode |
| --- | --- | --- |
| robotics or control world model | observation history, state, executed and planned actions | `repeated_transition` |
| video future prediction | prior frames, time/position, optional controls | `repeated_transition`, or `terminal_readout` for one-shot representation use |
| industrial or scientific dynamics | sensor history, controls, boundary conditions | `repeated_transition` |
| multi-agent or graph dynamics | node/edge states, interactions, actions, time | usually `repeated_transition` |
| predictive state for a downstream task | history, one future target, a classifier/regressor/retriever | `terminal_readout` |
| prediction-coordinate analysis | trajectories, probes, swaps, ablations, interventions | `factor_analysis` |
| existing world-model repository | dataset, model, rollout, evaluation, and split code | preserve the source task and generate a JEPA Anything interface |

Ordinary static classification or regression without world-state, future, or
dynamics intent should not activate this Skill.

## What “automatic invocation” means

After installation, `agents/openai.yaml` permits Codex to select this Skill from
the meaning of the request. Users normally do not need to mention JEPA or type
`$jepa-anything-skill`.

Automatic invocation is not a background service and cannot guarantee routing
for every ambiguous request. It means:

- Codex may load the Skill for world models, latent dynamics,
  action-conditioned prediction, rollout, planning, or control tasks;
- after loading, the Skill performs the JEPA Anything design instead of asking
  the user to author JEPA fields;
- explicit `$jepa-anything-skill` invocation remains the reliable fallback when
  a request might match several workflows;
- the standalone validator consumes an existing JSON/YAML design; it does not
  translate arbitrary natural language by itself.

## Install it in Codex

### Option 1: development symlink

From the repository root:

```bash
SKILLS_DIR="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$SKILLS_DIR"
ln -s "$(pwd)/jepa-anything-skill" \
  "$SKILLS_DIR/jepa-anything-skill"
```

Repository edits are immediately visible through the link. The command safely
fails when the destination exists; inspect the existing installation instead
of overwriting it.

### Option 2: packaged copy

Copy the complete directory to:

```text
${CODEX_HOME:-$HOME/.codex}/skills/jepa-anything-skill/
```

Do not copy only `SKILL.md`. The `agents/`, `scripts/`, `references/`, and
`assets/` directories are part of the Skill. Start a new Codex task or refresh
Skill discovery after installation.

### Installation smoke prompt

```text
Use $jepa-anything-skill to convert “predict the next 8 robot states from the
previous 4 states and actions” into a JEPA Anything design. Explain the mapping
only; do not train.
```

A loaded Skill should return a context/target mapping, one use mode,
adapter/encoder and `d/K/r` proposals, a validation plan, and the no-training
boundary.

## Minimum information from the user

The user does not need to provide a configuration. Four causal facts should be
recoverable from the request or supplied repository:

| Required fact | What it establishes | Example |
| --- | --- | --- |
| system and instance | which context and target belong to one concrete object or trajectory | `episode_id`, `machine_id`, `scene_id` |
| information available at prediction time | historical observations, current state, and truly known actions or inputs | eight prior frames, current joints, selected control |
| future target and horizon | what is predicted, when it begins, and how far it extends | the next 16 states or next video segment |
| state consumer | one-shot readout, repeated transition, or coordinate analysis | MPC rollout, classification probe, coordinate ablation |

Splits, metrics, compute constraints, and output location are also useful, but
the Skill should recover them from an existing repository when possible.

## Decisions made automatically

The Skill proposes the JEPA-specific choices instead of asking the user to fill
them in:

| Proposed item | Decision basis |
| --- | --- |
| observation/token adapter | modality, sampling, missing values, masks, and temporal structure |
| context–target boundary | original prediction boundary and information availability |
| target descriptor | horizon, target position, query, and availability time |
| exogenous inputs | whether actions, plans, or forcing are known at prediction time |
| encoder family | vector, sequence, graph, image/video, audio, or multimodal structure |
| `d/K/r` | source latent width, task scale, and compute envelope, with `K*r=d` |
| use mode | one-shot readout, repeated state transition, or coordinate analysis |
| four losses | factor prediction, projector geometry, target-coordinate activity, online-encoder activity |
| two baselines | standard JEPA and unconstrained multihead with capacity-match plans |
| audits and evaluation | leakage, geometry, activity, capacity, and claim–experiment coverage |

These are inspectable proposals, not claims of optimal settings. They must be
recorded in the configuration as compiler assumptions.

## Three input patterns

### 1. Natural-language task only

```text
I have robot trajectories with camera frames, joint state, executed actions,
and episode_id. Use the previous 8 observations and the action selected at each
step to roll out the next 16 states for model-predictive control. Episodes must
not cross data splits. Convert this into JEPA Anything, choose the remaining
design, and generate configuration and skeleton files. Do not train.
```

### 2. Convert an existing repository

```text
Inspect this repository's dataset loading, model inputs and outputs, rollout,
metrics, and split code. Preserve the world-model task and convert its interface
to JEPA Anything. Generate the design, validation report, and skeleton in a new
output directory; do not overwrite the implementation or start training. Show
the source-to-JEPA mapping and every unresolved assumption.
```

The Skill should inspect first and ask only for facts that cannot be recovered.

### 3. Force explicit invocation

```text
Use $jepa-anything-skill to convert the following world-model task into a
validated JEPA Anything design and no-training code skeleton: ...
```

## Automatic conversion flow

```text
natural-language task or existing repository
                    |
                    v
recover system, instance, inputs, actions, target, horizon, consumer, metrics
                    |
          missing causal blocker? ---- yes ----> ask only the necessary question
                    |
                    v
propose adapter, encoder, d/K/r, losses, baselines, and audits
                    |
                    v
write schema 2.0 design -> deterministic validation -> correct every failure
                                                        |
                                                        v
                                      generate skeleton + run contract tests
                                                        |
                                                        v
                              report mapping, paths, and unresolved assumptions
```

The Skill must preserve the source dataset, target, horizon, action
availability, split policy, metric, and downstream objective. JEPA Anything
changes the state interface and validation contract, not the user's task.

## When the Skill asks a question

A question is blocking only when guessing could change the task or create
target leakage, such as unknown instance identity, unknown future-input
availability, an unspecified target/horizon, or an ambiguous state consumer.

Adapter, encoder, OPF, `d/K/r`, initial loss weights, baseline structure, and
audits are not user blockers. The Skill proposes reversible defaults and stores
them under `task.metadata.compiler_assumptions`.

## Deliverables

```text
generated/
  config/design.json         validated schema 2.0 task design
  validation-report.json     deterministic checks; errors is zero on success
  scaffold-manifest.json     generated paths and SHA-256 digests
  pyproject.toml             minimal generated package configuration
  src/jepa_task/
    adapter.py               observation/token adapter interface
    model.py                 encoder, predictor, OPF, and synthesis interfaces
    activity.py              projected-target and online-encoder activity
    capacity.py              parameter, predictor-FLOP, and step audits
    usage.py                 selected downstream consumer
    evaluation.py            metrics, experiments, baselines, and claim coverage
    contracts.py             runtime invariants rendered from the design
  tests/test_contract.py      immediately runnable generated contract tests
```

The response should also include the source-to-JEPA conversion map, recovered
facts, proposed choices, validator and generated-test results, unresolved
data-owner assumptions, and an explicit statement that no training or empirical
claim was produced.

## Completion criteria

The design stage is complete only when:

- `validation-report.json` has `valid: true` and zero errors;
- system identity and temporal/field leakage checks pass;
- `K*r=d` and both coordinate packing and complete-state synthesis are defined;
- standard-JEPA and unconstrained-multihead baselines have capacity-match plans;
- every planned claim maps to matching experiments, metrics, baselines, and audits;
- generated contract tests pass;
- unresolved assumptions are visible rather than presented as facts.

This establishes a completed JEPA Anything task/interface conversion, not a
trained or empirically validated model.

## Use the command-line tools directly

The CLI is useful for CI, hand-authored designs, or inspecting a Skill output.
Run commands from the repository root.

### Validate the maintained example

```bash
mkdir -p work/quickstart

python3 jepa-anything-skill/scripts/validate_design.py \
  recipes/synthetic-linear-dynamics/design.expected.json \
  --output work/quickstart/validation-report.json \
  --pretty
```

Exit status is `0` for a valid design, `1` for contract errors, and `2` for a
parse, dependency, or I/O error. Each failed check includes a stable code, JSON
path, and explanation. Correct every error before generation.

### Generate a project skeleton

```bash
python3 jepa-anything-skill/scripts/generate_scaffold.py \
  recipes/synthetic-linear-dynamics/design.expected.json \
  --output-dir work/quickstart/generated-task \
  --config-format json
```

The destination must be missing or empty. The generator refuses invalid input,
undeclared output formats, and non-empty destinations.

### Run the generated contract tests

```bash
python3 -m unittest discover \
  -s work/quickstart/generated-task/tests \
  -v
```

These tests check the rendered contract and interfaces. They do not provide
data, implement adapters, measure model capacity, or execute optimization.

## Choose exactly one use mode

| Mode | Choose it when | Required design detail |
|---|---|---|
| `terminal_readout` | one final state feeds a classifier, regressor, or other readout | readout tasks and metrics |
| `repeated_transition` | the state is advanced repeatedly | sorted rollout horizons and whether each step needs exogenous input |
| `factor_analysis` | coordinates will be probed or intervened upon | probe plan, intervention plan, and prediction-coordinate-only reporting |

Do not combine the three modes in one design. Create separate designs when the
same data will support different consumers.

## Configuration contract

The canonical schema is documented in
[`jepa-anything-skill/references/config-contract.md`](jepa-anything-skill/references/config-contract.md).
The design procedure and decision criteria are in
[`jepa-anything-skill/references/design-protocol.md`](jepa-anything-skill/references/design-protocol.md).
The source-world-model conversion rules are in
[`jepa-anything-skill/references/world-model-conversion.md`](jepa-anything-skill/references/world-model-conversion.md).

Important invariants include:

- context and target share the same system and instance identity keys;
- context ends before the target starts;
- adapter fields and conditioning inputs have declared lineage;
- target fields cannot enter context through an undeclared path;
- `K * r == d` and complete state synthesis uses a full-rank analysis map;
- all factor and online-encoder coordinates have activity coverage;
- both required baseline families have parameter, FLOP, and step contracts;
- every planned claim is matched exactly to experiments, metrics, horizons,
  baselines, and audits;
- factor IDs remain neutral: `pc_000`, `pc_001`, and so on.

## Optional core library

Install the support library when implementing or testing tensor operations:

```bash
python3 -m pip install -e './jepa-anything-core[dev]'
```

It provides OPF decomposition/synthesis, loss components, capacity-matched
baseline helpers, EMA target updates, and numerical geometry/activity audits.
See [`jepa-anything-core/README.md`](jepa-anything-core/README.md) for its API.

## Repository checks

Run the complete deterministic check suite before publishing a Skill change:

```bash
make check
```

Individual commands are listed by `make help`. The complete check covers JSON
syntax, Python compilation, core and Skill tests, the maintained design,
example-task audit, and artifact manifest integrity.

## Troubleshooting

- **`python: command not found`** — use `python3`, as shown above.
- **A world-model request did not select the Skill** — make the future-state,
  action-conditioned, rollout, planning, or control intent explicit, or invoke
  `$jepa-anything-skill` directly.
- **The Skill asks the user for JEPA parameters first** — this is not the
  intended workflow. The user supplies only causal task facts; adapter,
  encoder, `d/K/r`, losses, baselines, and audits are Skill proposals.
- **Repository conversion repeats facts already in code** — instruct it to
  inspect dataset, input/output, rollout, metric, and split implementations
  before asking only for unrecoverable causal facts.
- **YAML dependency error** — install PyYAML or use JSON.
- **Validation exits with 1** — inspect `checks` entries whose `status` is
  `fail`; the `path` points to the exact field to correct.
- **`K * r != d`** — change the dimensions so the factors form one complete
  state interface.
- **Target leakage diagnostic** — correct the window, source cutoff, field
  allowlist, or conditioning lineage; do not silence the check.
- **Scaffold destination is non-empty** — choose a new directory. The generator
  will not overwrite existing work.
- **Claim coverage failure** — make claim and experiment metric, direction,
  split, horizons, baselines, audits, use mode, and reciprocal IDs identical.
- **Generated test requests measurements** — supply real geometry, activity,
  and capacity measurements from the later implementation; configured budgets
  are planning values, not measured evidence.

## Boundary between proposal and evidence

```text
natural-language task
        |
        v
LLM design proposal ---------> reviewable JSON/YAML
                                  |
                                  v
                         deterministic validator
                                  |
                    fail <--------+--------> pass
                    reasons                   |
                                              v
                                      code skeleton
                                              |
                                      explicit user-run
                                        experiments
                                              |
                                              v
                                  audits + claim evidence
```

The generator stops at the skeleton boundary.  Training, checkpoint download,
and publication claims are separate, explicit actions.

## Skill package

`jepa-anything-skill/` is the primary entry point. The core package supplies
deterministic mechanisms referenced by generated implementations, the recipe is
a maintained end-to-end example, and the checkpoint manifest defines how a
future artifact can be described without being bundled into the Skill. Expected
checks remain plans until an execution workflow supplies measured evidence.

## Contributing

Changes that weaken a deterministic gate must include a concrete threat model
and regression test.  See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0.  See [LICENSE](LICENSE).
