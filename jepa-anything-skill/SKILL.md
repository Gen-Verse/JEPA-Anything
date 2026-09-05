---
name: jepa-anything-skill
description: Automatically convert a natural-language world-model task into a validated JEPA Anything design and inspectable, no-training code skeleton. Use whenever a user provides, designs, or asks to adapt a world model, latent-dynamics model, action-conditioned predictor, predictive-state model, or rollout task—even when they do not mention JEPA or this Skill. Do not use it for ordinary static prediction with no world-state intent, model training, or semantic naming of prediction coordinates.
---

# JEPA Anything World-Model Compiler

Turn the user's existing world-model intent into a verifiable JEPA Anything world-state interface. The user supplies the domain task in their own language; do not require them to know JEPA, OPF, context-target terminology, or `d`, `K`, and `r`. Deliver configuration and inspectable interfaces, never a hidden training run.

## Trigger and outcome

Invoke this Skill implicitly when a request is about learning or adapting a world model, predicting future latent or observed state, action-conditioned dynamics, multi-step rollout, planning/control dynamics, or converting an existing world-model task or repository. Explicit `$jepa-anything-skill` invocation remains supported but is not required.

The conversion target is always:

```text
raw world-model task
  -> normalized causal prediction boundary
  -> JEPA Anything design proposal
  -> deterministic validation
  -> configuration + inspectable code skeleton
```

Do not merely explain how the user could perform the conversion. When the request authorizes creating or adapting the task, produce the design files and skeleton within the requested workspace.

## Workflow

1. Read [references/world-model-conversion.md](references/world-model-conversion.md). Extract the original task's system, observations, actions or other known inputs, prediction target/horizon, intended rollout or consumer, data split, and metrics. If a repository or configuration is supplied, inspect it rather than asking the user to restate facts already present.
2. Separate missing facts into:
   - **blocking causal facts**: system-instance identity, what is available at prediction time, target and horizon, or intended consumer when it cannot be inferred;
   - **JEPA design choices**: adapter, encoder, state dimensions, coordinate grouping, losses, baselines, and audits.
   Ask only for genuinely blocking causal facts. Propose reversible defaults for JEPA design choices and record them under `task.metadata.compiler_assumptions`; never make the user design JEPA for you.
3. Read [references/design-protocol.md](references/design-protocol.md) and compile the task into one causal context-target boundary. Select exactly one of `terminal_readout`, `repeated_transition`, or `factor_analysis` from the original downstream intent.
4. Read [references/config-contract.md](references/config-contract.md) before authoring or migrating configuration. Emit only canonical coordinate IDs (`pc_000`, `pc_001`, …); never assign them domain semantics.
5. Propose the observation/token adapter, context and target encoder family, target descriptors, known exogenous inputs, `d/K/r`, learned projectors, an orthogonality strategy, factor-coordinate concatenation, full-state pseudoinverse synthesis, four explicit losses, and both capacity-matched baselines. Default to `soft_gram`; use `qr_init` or post-optimizer `qr_retraction` only as an explicit design choice. Never insert QR into the forward path. Treat all design choices as hypotheses.
6. Express every planned claim as a structured evidence contract. Metric, direction, split, horizons, baselines, audits, experiments, model, and use mode must match exactly.
7. Write the proposed configuration, run the deterministic validator, and correct every design error:

   ```bash
   python3 scripts/validate_design.py path/to/design.json --pretty
   ```

8. Generate the skeleton only from a valid design, then run its generated contract tests:

   ```bash
   python3 scripts/generate_scaffold.py path/to/design.json --output-dir path/to/output
   python3 -m unittest discover -s path/to/output/tests -v
   ```

9. Report the mapping from the original world-model task to JEPA Anything, the generated paths, validator result, selected use mode, proposed adapter/encoder and dimensions, and unresolved data-owner assumptions.

The generator refuses invalid designs and non-empty destinations. Its adapter, OPF state, geometry/activity audits, mode consumer, capacity audit, claim audit, and contract tests contain no optimizer or training loop.

## Non-negotiable boundaries

- Context and target share `task.system_id` and exactly the same non-empty `identity_keys`. A joint system must be declared explicitly.
- Bind adapter context fields to declared context-source lineage. Reject temporal overlap, target-derived context, unavailable descriptors, and unavailable exogenous inputs.
- Require `K * r == d`, canonical factor-coordinate concatenation, `learned_projectors`, and complete-state synthesis through the Moore-Penrose pseudoinverse of a measured full-rank analysis map.
- Permit `soft_gram`, one-time `qr_init`, or sign-canonicalized `qr_retraction` after optimizer steps. QR modes do not waive projector-Gram, rank, conditioning, synthesis, or capacity audits.
- Require factor MSE, within/cross projector-Gram loss, a standard-deviation floor for every projected factor dimension, and a standard-deviation floor for every online encoder dimension.
- Require both `standard_jepa` and `unconstrained_multihead` baselines to declare and later measure full-model trainable parameters, predictor FLOPs, and identical steps under separate parameter/FLOPs tolerances.
- A passing validator proves only that the declared design and evidence plan are internally complete. It does not prove activity, synthesis quality, semantics, model quality, or claim support.
- Do not train, download checkpoints, or report empirical results. Hand training requests to a separate execution workflow after the validated design is complete.

## Resources

- [references/world-model-conversion.md](references/world-model-conversion.md) maps ordinary world-model requests to JEPA Anything choices and explains safe automatic defaults.
- [references/config-contract.md](references/config-contract.md) is the normative schema 2.0 and diagnostic reference.
- [references/design-protocol.md](references/design-protocol.md) explains design decisions and mathematical semantics.
- `scripts/validate_design.py` performs deterministic validation; `scripts/generate_scaffold.py` renders `assets/scaffold/`.
