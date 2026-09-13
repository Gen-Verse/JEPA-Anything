[简体中文](README.zh-CN.md) · [Quick start](#quick-start) · [Core library](jepa-anything-core/README.md) · [Example recipe](recipes/synthetic-linear-dynamics/README.md)

<p align="center">
  <img src="assets/figures/framework-overview.png" alt="JEPA-Anything overview: cross-domain predictive learning and Orthogonal Predictive Factorization" width="100%" />
</p>

## At a glance

World models learn internal states to infer unobserved, intervened, or future states from the current context. Cells, molecules, physical fields, control environments, and clinical trajectories differ radically, yet share this predictive problem: **can different worlds be modeled through a common predictive learning principle?**

**JEPA-Anything is a domain-agnostic framework for learning predictive models of different worlds.** Building on joint-embedding predictive architectures (JEPAs), it replaces a monolithic target representation with **Orthogonal Predictive Factorization (OPF)**. OPF organizes a latent world state into complementary predictive factors, learns each through a dedicated prediction pathway, and recombines them into a complete state for downstream readout, intervention prediction, planning, and multi-step rollout.

Different domains retain their observations, context–target construction, and encoders while sharing the predictive core and latent world-state interface. The commonality is the learning principle and interface; domains need not share a single encoder or one set of model weights.

## Why JEPA-Anything?

| Advantage | How it works | What it enables |
|---|---|---|
| One principle across worlds | Domain-specific inputs connect to a shared factorized predictive core | Reuse of the modeling approach while preserving domain structure |
| Complementary prediction pathways | Factors are learned through dedicated pathways and jointly synthesize the state | A structured interface for representation reuse, intervention, and out-of-distribution prediction |
| A complete state for continued use | Predicted factors are recombined for downstream consumption | Readout, planning, and long-horizon rollout |
| Prediction connected to scientific analysis | Domain experiments relate predictive modes to the underlying system | Analysis linked to biological interventions and physical laws |

## Project map

This repository provides the reusable core, task-design tools, and an executable structural example. It is designed as a self-contained starting point for implementing predictive-factor workflows in a new domain.


| Component | Purpose | Start here |
|---|---|---|
| Core | Reusable projection, objectives, baselines, and representation diagnostics | [Library guide](jepa-anything-core/README.md) |
| Task design | Validate a task specification and generate an implementation scaffold | [Design contract](jepa-anything-skill/references/config-contract.md) |
| Example | Follow a controlled dynamical system from specification to structural checks | [Synthetic linear dynamics](recipes/synthetic-linear-dynamics/README.md) |
| Architecture | Understand the relationship between adapters, the predictive core, and downstream use | [Architecture guide](docs/architecture.md) |
| Artifact metadata | Describe and verify model artifact provenance | [Manifest guide](checkpoints/README.md) |

## Quick start

<p align="center">
  <b>⚙️ Install</b>&nbsp;&nbsp;•&nbsp;&nbsp;<b>🧪 Run</b>&nbsp;&nbsp;•&nbsp;&nbsp;<b>🧩 Design</b>
</p>

The commands below use a Linux/macOS shell. Clone the repository using an authorized GitHub account, then run the remaining commands from its root directory. Python 3.10 or newer is required; the core uses PyTorch, while the design tools and structural example use the Python standard library.

### 1. ⚙️ Install the core

```bash
git clone https://github.com/Gen-Verse/JEPA-Anything.git
cd JEPA-Anything
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e './jepa-anything-core[dev]'
```

### 2. 🧪 Run the structural example

```bash
python3 recipes/synthetic-linear-dynamics/run_recipe.py --quiet
```

This checks the example's data and interface consistency. It does not train a model or produce benchmark performance.

### 3. 🧩 Validate and generate a task scaffold

```bash
python3 jepa-anything-skill/scripts/validate_design.py \
  recipes/synthetic-linear-dynamics/design.expected.json \
  --pretty --output work/quickstart/validation-report.json

python3 jepa-anything-skill/scripts/generate_scaffold.py \
  recipes/synthetic-linear-dynamics/design.expected.json \
  --output-dir work/quickstart/generated-task --config-format json

python3 -m unittest discover \
  -s work/quickstart/generated-task/tests -p 'test_*.py' -v
```

The generator requires an empty or new output directory. For repeated runs, choose a new output directory and update the test path accordingly. The report records validation results; the generated directory contains the task configuration, interfaces, and contract tests.

<details>
<summary><b>From observations to predictive factors</b></summary>

An encoder maps observations into a latent state. Factor projections organize that state into groups of predictive coordinates. A predictor models their changes under the available context or actions, and the downstream task consumes the resulting state through prediction, rollout, or analysis.

The core supplies the common operations; each domain supplies its observation adapter and task-specific evaluation. A factor's physical or biological interpretation is established through experiments, rather than its index.

</details>

<details>
<summary><b>From a task description to implementation</b></summary>

The task-design tools record which observations are available, what future target is predicted, and how predictions will be used. A deterministic validator checks the specification before a scaffold is generated. The scaffold provides interfaces for adapters, models, diagnostics, and evaluation; domain implementation completes those interfaces.

</details>

## Scenario atlas

<p align="center">
  <img src="assets/figures/scenario-atlas.png" alt="JEPA-Anything scenario atlas spanning terminal readout, latent world dynamics, and scientific analysis" width="100%" />
</p>

The atlas gives a compact map of the project’s use cases: terminal readout in visual, single-cell, and clinical settings; intervention, out-of-distribution, planning, and molecular rollouts; and scientific analyses that connect learned factors to wet-lab experiments and physical laws.

## Repository scope

The figures illustrate the broader research scenarios. This repository contains the reusable core, task-design tools, and a synthetic structural example; it does not include the domain datasets or trained model weights depicted in the atlas. The [checkpoint manifest](checkpoints/manifest.json) records an untrained example with no performance claims. Quantitative and experimental labels in the figures require corresponding study evidence; the structural checks do not reproduce those results.

## Development checks

After installing the development dependencies:

```bash
make check
```

This runs source checks, core and task-design tests, reference-design validation, the structural recipe, and manifest verification. Use `make help` to see individual targets.

## License

See [LICENSE](LICENSE) and the [core license](jepa-anything-core/LICENSE).
