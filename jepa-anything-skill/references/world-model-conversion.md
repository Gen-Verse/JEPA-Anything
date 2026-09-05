# World-model task to JEPA Anything

Use this reference when the input is an ordinary world-model request, an existing task description, or a repository that was not written for JEPA Anything. The user is not expected to supply JEPA-specific fields.

## 1. Preserve the original task

First restate the source task without changing its scientific or product objective:

```text
system and instance -> available history -> known action/input -> future target
                    -> intended consumer -> evaluation
```

JEPA Anything changes the predictive state interface and evidence contract. It must not silently change the dataset, target, horizon, action availability, downstream goal, split policy, or metric. Put a compact source-task summary and every provisional choice in `task.metadata.compiler_assumptions`.

When a repository is supplied, inspect its dataset schema, sampling code, model inputs/outputs, rollout loop, evaluation code, and split logic. Prefer those facts over guesses or a new questionnaire.

## 2. Ask only about causal blockers

Continue automatically when the task establishes all four of these facts:

1. what concrete system instance a context and target belong to;
2. what information is available when prediction begins;
3. what future observation or latent target is predicted, and at what horizon;
4. how the resulting state is consumed.

Ask a short question only when one of these cannot be recovered and a guess could create target leakage or change the task. Do not ask the user to choose OPF, losses, baseline structure, canonical coordinate IDs, or JEPA-specific schema fields.

Unknown non-causal implementation choices are not blockers. Select a provisional, internally valid design, label it as a proposal, and expose it in the output configuration.

## 3. Compile the prediction boundary

Map the source task into these fields:

| Source world-model concept | JEPA Anything field |
| --- | --- |
| environment, plant, scene, patient trajectory, market, simulator | `task.system_id` |
| episode, device, trajectory, subject, scene, sequence | `identity_keys` |
| past frames, sensor history, current state, observation tokens | `observation.context` and adapter context allowlist |
| next frame, future sensors, next latent, future state | `observation.target` and adapter target allowlist |
| horizon index, query time, target mask or location | target descriptor |
| action sequence, control plan, known forcing, static metadata | exogenous input only if available at prediction time |
| rollout/model-predictive control | `repeated_transition` |
| one-shot downstream probe or retrieval | `terminal_readout` |
| coordinate probing, interventions, disentanglement analysis | `factor_analysis` |

For action-conditioned models, distinguish executed past actions in context from planned future actions. A future action may condition prediction only when it is actually known at that step. If the policy will choose actions during rollout, the transition interface may accept the chosen action after it becomes available; do not encode unchosen future actions as observed context.

## 4. Propose the adapter and encoder

Preserve the modality and sampling semantics of the source task:

| Data | Adapter proposal | Encoder proposal |
| --- | --- | --- |
| vector state or tabular trajectory | one token per time step, feature projection, time/field masks | MLP for Markov state; temporal Transformer or TCN for history |
| image sequence or video | patch or tubelet tokens with spatial-temporal positions | ViT or video Transformer |
| audio or dense waveform | framed spectral or learned temporal tokens | conformer, temporal Transformer, or convolutional encoder |
| graph or interacting agents | node/edge tokens with instance and time identity | graph network or graph Transformer |
| multimodal trajectory | modality-specific adapters into one width, plus modality/time tags | multimodal Transformer with explicit missing-modality masks |

The context and target encoders must emit the same state width `d`. Reuse the source encoder family when adapting an existing project unless doing so violates the shared-state interface. Record normalization, missing-value handling, temporal positions, padding masks, and field allowlists in the adapter proposal.

## 5. Select `d`, `K`, and `r` automatically

Honor a source model's latent width when it is known and can be factored. Otherwise choose a provisional capacity tier:

| Task scale | Default proposal |
| --- | --- |
| low-dimensional state, control, or sensor task | `d=64`, `K=4`, `r=16` |
| image, video, audio, graph, or medium multimodal task | `d=256`, `K=8`, `r=32` |
| large multimodal task with an existing wide encoder | use the existing width or `d=512`, normally `K=8`, `r=64` |

If an existing width is not conveniently factorable, prefer a nearby interface width supported by an explicit projection adapter. Always enforce `K*r=d`. These values are capacity proposals, not empirically optimal choices. Record the reason and compute assumption.

Use `soft_gram` as the default orthogonality strategy. Treat `qr_init` and
post-optimizer `qr_retraction` as explicit alternatives when exact initialization
or a hard-orthogonality ablation is useful; account for retraction cost and never
place QR inside the prediction forward path.

## 6. Preserve the source evaluation objective

Use the original metric whenever it measures the unchanged target and consumer. Add JEPA Anything audits; do not replace the task metric with an audit metric.

- A one-step source task uses its original held-out prediction or downstream metric.
- A rollout task evaluates every declared rollout horizon, not only the final step.
- A control task separates state-prediction metrics from downstream return, success, safety, or constraint metrics.
- A factor-analysis task needs probes, swaps, ablations, or interventions; prediction error alone cannot establish coordinate semantics.

Always add standard JEPA and unconstrained multihead baselines under matched full-model parameters, predictor FLOPs, and training steps. Planned counts may be estimates at design time, but generated audits must require measured counts before supporting a claim.

## 7. Deliver a conversion map

Alongside the validated configuration and skeleton, summarize:

```text
original context          -> JEPA context tokens
original prediction       -> target encoder and projected coordinates
original action/input     -> exogenous conditioning and availability rule
original latent/state     -> d-dimensional JEPA state
original rollout/readout  -> selected use mode
original evaluation       -> experiments, metrics, baselines, and audits
```

Clearly separate recovered source-task facts, proposed JEPA choices, blocking unresolved facts, and later empirical measurements. A valid configuration is a safe handoff, not evidence that the converted model will outperform the source world model.
