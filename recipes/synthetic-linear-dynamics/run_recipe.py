#!/usr/bin/env python3
"""Run a dependency-free structural audit for the synthetic JEPA recipe.

This script intentionally does not train a model. It generates a small seeded
system in memory, checks task-interface invariants, and writes a JSON diagnostic
report suitable for CI or for reviewing a generated design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RECIPE = SCRIPT_DIR / "recipe.json"
DEFAULT_DESIGN = SCRIPT_DIR / "design.expected.json"


class Audit:
    """Collect pass/fail checks without stopping at the first error."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def add(self, check_id: str, passed: bool, detail: str, **evidence: Any) -> None:
        item: Dict[str, Any] = {
            "id": check_id,
            "passed": bool(passed),
            "detail": detail,
        }
        if evidence:
            item["evidence"] = evidence
        self.checks.append(item)

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_shape(value: Any) -> Tuple[int, int] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(row, list) for row in value):
        return None
    widths = {len(row) for row in value}
    if len(widths) != 1:
        return None
    return len(value), next(iter(widths))


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(coef * value for coef, value in zip(row, vector)) for row in matrix]


def population_variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / len(values)


def _hash_float(digest: "hashlib._Hash", value: float) -> None:
    # Big-endian binary64 makes the in-memory generated-data fingerprint
    # independent of locale and JSON float formatting.
    digest.update(struct.pack(">d", float(value)))


def generate_activity_evidence(recipe: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = recipe["provenance"]
    system = recipe["system"]
    data = recipe["data"]
    transition = system["transition"]
    observation = system["observation"]
    split = data["split"]
    representation = recipe["representation"]

    seed = int(provenance["seed"])
    rng = random.Random(seed)
    state_matrix = transition["state_matrix"]
    control_vector = [row[0] for row in transition["control_matrix"]]
    observation_matrix = observation["matrix"]
    process_noise = float(transition["process_noise_std"])
    observation_noise = float(observation["noise_std"])
    trajectory_count = int(data["trajectory_count"])
    steps = int(data["steps_per_trajectory"])
    train_first, train_last = split["train_ids"]
    factor_count = int(representation["K"])
    factor_width = int(representation["r"])

    all_factor_values: List[List[List[float]]] = [
        [[] for _ in range(factor_width)] for _ in range(factor_count)
    ]
    fingerprint = hashlib.sha256()
    fingerprint.update(b"jepa-anything.synthetic-linear-dynamics/v1\0")

    for trajectory_id in range(trajectory_count):
        phase = rng.uniform(-math.pi, math.pi)
        state = [rng.gauss(0.0, 0.6) for _ in range(int(system["state_dim"]))]
        for step in range(steps):
            control = 0.7 * math.sin(0.11 * step + phase) + 0.2 * math.cos(0.037 * step)
            projected = matvec(observation_matrix, state)
            observed = [value + rng.gauss(0.0, observation_noise) for value in projected]

            fingerprint.update(struct.pack(">II", trajectory_id, step))
            for value in state:
                _hash_float(fingerprint, value)
            _hash_float(fingerprint, control)
            for value in observed:
                _hash_float(fingerprint, value)

            if train_first <= trajectory_id <= train_last:
                for factor_index in range(factor_count):
                    for coordinate_index in range(factor_width):
                        all_factor_values[factor_index][coordinate_index].append(
                            state[factor_index * factor_width + coordinate_index]
                        )

            deterministic_next = matvec(state_matrix, state)
            state = [
                deterministic_next[index]
                + control_vector[index] * control
                + rng.gauss(0.0, process_noise)
                for index in range(len(state))
            ]

    factors: List[Dict[str, Any]] = []
    for factor_index, coordinate_values in enumerate(all_factor_values):
        coordinate_variances = [population_variance(values) for values in coordinate_values]
        coordinate_stddevs = [math.sqrt(value) for value in coordinate_variances]
        factors.append(
            {
                "id": f"pc_{factor_index:03d}",
                "coordinate_variances": coordinate_variances,
                "coordinate_stddevs": coordinate_stddevs,
                "minimum_coordinate_stddev": min(coordinate_stddevs),
                "mean_coordinate_variance": math.fsum(coordinate_variances)
                / len(coordinate_variances),
                "sample_count_per_coordinate": len(coordinate_values[0]),
            }
        )

    return {
        "generator_algorithm": provenance["generator_algorithm"],
        "seed": seed,
        "trajectory_count": trajectory_count,
        "steps_per_trajectory": steps,
        "generated_stream_sha256": fingerprint.hexdigest(),
        "factors": factors,
    }


def audit_recipe(recipe: Mapping[str, Any], design: Mapping[str, Any]) -> Audit:
    audit = Audit()

    audit.add(
        "recipe_schema",
        recipe.get("schema_version") == "jepa-anything.recipe/v1",
        "Recipe declares the supported source schema.",
        observed=recipe.get("schema_version"),
        expected="jepa-anything.recipe/v1",
    )

    system = recipe.get("system", {})
    transition = system.get("transition", {})
    observation = system.get("observation", {})
    state_dim = system.get("state_dim")
    control_dim = system.get("control_dim")
    observation_dim = system.get("observation_dim")
    a_shape = matrix_shape(transition.get("state_matrix"))
    b_shape = matrix_shape(transition.get("control_matrix"))
    h_shape = matrix_shape(observation.get("matrix"))
    shapes_valid = (
        isinstance(state_dim, int)
        and isinstance(control_dim, int)
        and isinstance(observation_dim, int)
        and a_shape == (state_dim, state_dim)
        and b_shape == (state_dim, control_dim)
        and h_shape == (observation_dim, state_dim)
    )
    audit.add(
        "system_matrix_shapes",
        shapes_valid,
        "Transition, control, and observation matrices match declared dimensions.",
        state_matrix_shape=a_shape,
        control_matrix_shape=b_shape,
        observation_matrix_shape=h_shape,
    )

    task = recipe.get("task", {})
    context = task.get("context", {})
    target = task.get("target", {})
    expected_identity = system.get("sample_identity_keys")
    same_system = (
        bool(system.get("system_id"))
        and task.get("system_id") == system.get("system_id")
        and context.get("system_id") == system.get("system_id")
        and target.get("system_id") == system.get("system_id")
        and context.get("identity_keys") == expected_identity
        and target.get("identity_keys") == expected_identity
        and "trajectory_id" in (expected_identity or [])
    )
    audit.add(
        "same_underlying_system",
        same_system,
        "Context and target share explicit system and trajectory identity.",
        system_id=system.get("system_id"),
        identity_keys=expected_identity,
    )

    adapter = task.get("adapter", {})
    source_entries = context.get("sources", [])
    sources_by_id = {
        source.get("id"): source
        for source in source_entries
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    }
    declared_context_fields = {
        field
        for source in sources_by_id.values()
        for field in source.get("fields", [])
        if isinstance(field, str)
    }
    context_input_fields = adapter.get("context_input_fields", [])
    target_input_fields = adapter.get("target_input_fields", [])
    target_fields = target.get("fields", [])
    context_end = max(context.get("observation_offsets", [0]))
    sources_available = bool(sources_by_id) and all(
        isinstance(source.get("available_through"), int)
        and source["available_through"] <= context_end
        for source in sources_by_id.values()
    )

    def lineage_is_available(item: Mapping[str, Any], allowed_types: set[str]) -> bool:
        lineage = item.get("lineage", {})
        if not isinstance(lineage, Mapping):
            return False
        source_type = lineage.get("source_type")
        available_through = lineage.get("available_through")
        source_id = lineage.get("source_id")
        if (
            source_type not in allowed_types
            or not isinstance(source_id, str)
            or not isinstance(available_through, int)
            or available_through > context_end
        ):
            return False
        if source_type == "context_source":
            source = sources_by_id.get(source_id, {})
            return item.get("name") in source.get("fields", [])
        if source_type == "external_plan":
            return source_id in sources_by_id
        return True

    descriptor_fields = task.get("target_descriptor", {}).get("fields", [])
    exogenous_fields = task.get("exogenous_inputs", {}).get("fields", [])
    lineage_valid = (
        isinstance(context_input_fields, list)
        and bool(context_input_fields)
        and set(context_input_fields).issubset(declared_context_fields)
        and isinstance(target_input_fields, list)
        and target_input_fields == target_fields
        and set(target.get("source_ids", [])).issubset(sources_by_id)
        and sources_available
        and all(
            isinstance(item, Mapping)
            and lineage_is_available(item, {"context_source", "prediction_request"})
            for item in descriptor_fields
        )
        and all(
            isinstance(item, Mapping)
            and lineage_is_available(item, {"context_source", "external_plan"})
            for item in exogenous_fields
        )
    )
    audit.add(
        "context_input_lineage",
        lineage_valid,
        "Adapter, descriptor, and exogenous inputs have declared prediction-time lineage.",
        context_input_fields=context_input_fields,
        target_input_fields=target_input_fields,
        context_source_ids=sorted(sources_by_id),
        context_end=context_end,
    )

    context_offsets = context.get("observation_offsets", [])
    target_offsets = target.get("observation_offsets", [])
    offsets_numeric = (
        isinstance(context_offsets, list)
        and isinstance(target_offsets, list)
        and bool(context_offsets)
        and bool(target_offsets)
        and all(isinstance(value, int) for value in context_offsets + target_offsets)
    )
    offsets_safe = (
        offsets_numeric
        and max(context_offsets) <= 0
        and min(target_offsets) > 0
        and set(context_offsets).isdisjoint(target_offsets)
    )
    descriptor = task.get("target_descriptor", {})
    descriptor_fields = descriptor.get("fields", [])
    safe_descriptor_sources = {"relative_offset", "known_exogenous_input", "static_metadata"}
    descriptor_safe = (
        descriptor.get("reads_target_observation_values") is False
        and descriptor.get("reads_latent_state") is False
        and isinstance(descriptor_fields, list)
        and all(field.get("source") in safe_descriptor_sources for field in descriptor_fields)
        and all(
            field.get("availability") == "known_at_prediction_time"
            and field.get("role") == "index_only"
            for field in descriptor_fields
        )
    )
    normalization = recipe.get("data", {}).get("normalization", {})
    leakage_policy = task.get("leakage_policy", {})
    preprocessing_safe = (
        normalization.get("fit_on") == "train_only"
        and normalization.get("target_statistics_visible_to_context") is False
        and leakage_policy.get("split_by_trajectory") is True
        and leakage_policy.get("normalization_fit_on_train_only") is True
    )
    audit.add(
        "no_target_leakage",
        offsets_safe and descriptor_safe and preprocessing_safe,
        "Temporal selectors, descriptors, splits, and normalization block target leakage.",
        context_offsets=context_offsets,
        target_offsets=target_offsets,
        descriptor_safe=descriptor_safe,
        preprocessing_safe=preprocessing_safe,
    )

    representation = recipe.get("representation", {})
    d = representation.get("d")
    k = representation.get("K")
    r = representation.get("r")
    factors = representation.get("factors", [])
    analysis = representation.get("analysis", {})
    synthesis = representation.get("synthesis", {})
    composition_valid = (
        all(isinstance(value, int) and value > 0 for value in (d, k, r))
        and k * r == d
        and representation.get("coordinate_layout") == "concatenate"
        and analysis.get("kind") == "learned_projectors"
        and analysis.get("projector_count") == k
        and analysis.get("projector_shape") == [d, r]
        and analysis.get("target_stop_gradient") is True
        and synthesis.get("kind") == "moore_penrose_pseudoinverse"
        and synthesis.get("analysis_map") == "projector_transpose"
        and synthesis.get("require_full_rank") is True
        and isinstance(factors, list)
        and len(factors) == k
        and all(factor.get("width") == r for factor in factors)
        and sum(factor.get("width", 0) for factor in factors) == d
    )
    audit.add(
        "state_composition",
        composition_valid,
        "Factor coordinates concatenate to d and a distinct full-rank pseudoinverse synthesis is declared.",
        d=d,
        K=k,
        r=r,
        product=(k * r if isinstance(k, int) and isinstance(r, int) else None),
        coordinate_layout=representation.get("coordinate_layout"),
        projector_shape=analysis.get("projector_shape"),
        synthesis=synthesis,
        combined_analysis_map_shape=([d, d] if composition_valid else None),
    )

    semantic_policy_valid = all(
        factor.get("semantic_label") is None
        and factor.get("status") == "uninterpreted_predictive_coordinate"
        and factor.get("id") == f"pc_{index:03d}"
        for index, factor in enumerate(factors)
    )
    audit.add(
        "no_unvalidated_factor_names",
        semantic_policy_valid,
        "Factors use neutral coordinate identifiers and carry no semantic labels.",
        factor_ids=[factor.get("id") for factor in factors],
    )

    if shapes_valid and composition_valid:
        activity = generate_activity_evidence(recipe)
        activity_contract = representation.get("activity_audit", {})
        minimum = activity_contract.get("minimum_std")
        observed_coordinate_count = sum(
            len(factor["coordinate_stddevs"]) for factor in activity["factors"]
        )
        activity_valid = (
            isinstance(minimum, (int, float))
            and minimum > 0
            and activity_contract.get("metric") == "coordinate_std_floor"
            and activity_contract.get("required_coordinate_count") == d
            and len(activity["factors"]) == k
            and observed_coordinate_count == d
            and all(
                stddev >= minimum
                for factor in activity["factors"]
                for stddev in factor["coordinate_stddevs"]
            )
        )
    else:
        activity = {"factors": [], "not_run_reason": "invalid dimensions or composition"}
        minimum = representation.get("activity_audit", {}).get("minimum_std")
        observed_coordinate_count = 0
        activity_valid = False
    audit.add(
        "factor_activity",
        activity_valid,
        "Every synthetic oracle coordinate independently exceeds the configured standard-deviation floor.",
        minimum_coordinate_stddev=minimum,
        observed_coordinate_count=observed_coordinate_count,
        activity=activity,
    )

    losses = recipe.get("losses", {})
    factor_prediction = losses.get("factor_prediction", {})
    projector_orthogonality = losses.get("projector_orthogonality", {})
    factor_activity = losses.get("factor_activity", {})
    online_encoder_activity = losses.get("online_encoder_activity", {})
    expected_factor_ids = [f"pc_{index:03d}" for index in range(k)]
    expected_factor_coordinates = [
        {"factor_id": factor_id, "coordinate_index": coordinate_index}
        for factor_id in expected_factor_ids
        for coordinate_index in range(r)
    ]
    objective_contract_valid = (
        factor_prediction.get("enabled") is True
        and factor_prediction.get("kind") == "factor_mse"
        and isinstance(factor_prediction.get("weight"), (int, float))
        and factor_prediction["weight"] > 0
        and projector_orthogonality.get("enabled") is True
        and projector_orthogonality.get("kind") == "projector_gram"
        and set(projector_orthogonality.get("components", []))
        == {"within_projector", "cross_projector"}
        and projector_orthogonality.get("projectors") == expected_factor_ids
        and factor_activity.get("enabled") is True
        and factor_activity.get("kind") == "coordinate_std_floor"
        and factor_activity.get("scope") == "projected_target_factors"
        and factor_activity.get("coordinates") == expected_factor_coordinates
        and isinstance(factor_activity.get("min_std"), (int, float))
        and factor_activity["min_std"] > 0
        and isinstance(factor_activity.get("epsilon"), (int, float))
        and factor_activity["epsilon"] > 0
        and online_encoder_activity.get("enabled") is True
        and online_encoder_activity.get("kind") == "coordinate_std_floor"
        and online_encoder_activity.get("scope") == "online_context_encoder"
        and online_encoder_activity.get("dimensions") == list(range(d))
        and isinstance(online_encoder_activity.get("min_std"), (int, float))
        and online_encoder_activity["min_std"] > 0
        and isinstance(online_encoder_activity.get("epsilon"), (int, float))
        and online_encoder_activity["epsilon"] > 0
    )
    audit.add(
        "loss_contract",
        objective_contract_valid,
        "Recipe declares factor MSE, projector-Gram geometry, and coordinate-wise activity terms.",
        loss_keys=sorted(losses) if isinstance(losses, dict) else [],
        factor_coordinate_count=len(factor_activity.get("coordinates", [])),
        online_encoder_coordinate_count=len(online_encoder_activity.get("dimensions", [])),
    )

    baselines = recipe.get("baselines", [])
    by_family = {
        baseline.get("family"): baseline
        for baseline in baselines
        if isinstance(baseline, dict)
    }
    standard = by_family.get("standard_jepa", {})
    multihead = by_family.get("unconstrained_multihead", {})
    reference_match = representation.get("capacity", {})
    standard_match = standard.get("capacity_match", {})
    multihead_match = multihead.get("capacity_match", {})
    capacity_keys = {
        "full_model_trainable_parameters",
        "predictor_flops",
        "training_steps",
        "parameter_tolerance",
        "predictor_flops_tolerance",
        "parameter_scope",
        "flops_scope",
    }
    capacity_values_valid = (
        set(reference_match) == capacity_keys
        and reference_match.get("parameter_scope") == "full_trainable_model"
        and reference_match.get("flops_scope") == "predictor_forward_per_target"
        and isinstance(reference_match.get("full_model_trainable_parameters"), int)
        and reference_match["full_model_trainable_parameters"] > 0
        and isinstance(reference_match.get("predictor_flops"), int)
        and reference_match["predictor_flops"] > 0
        and isinstance(reference_match.get("training_steps"), int)
        and reference_match["training_steps"] > 0
        and isinstance(reference_match.get("parameter_tolerance"), (int, float))
        and 0 <= reference_match["parameter_tolerance"] < 1
        and isinstance(reference_match.get("predictor_flops_tolerance"), (int, float))
        and 0 <= reference_match["predictor_flops_tolerance"] < 1
    )

    def baseline_capacity_matches(value: Mapping[str, Any]) -> bool:
        return (
            all(value.get(key) == reference_match.get(key) for key in capacity_keys)
            and value.get("reference_model_id") == "opf"
            and value.get("encoder_policy") == "same_family_and_width"
            and value.get("data_policy") == "same_samples_and_augmentations"
        )

    baseline_valid = (
        standard.get("required") is True
        and multihead.get("required") is True
        and capacity_values_valid
        and baseline_capacity_matches(standard_match)
        and baseline_capacity_matches(multihead_match)
        and multihead.get("head_count") == k
        and multihead.get("head_dim") == r
    )
    audit.add(
        "capacity_matched_baselines",
        baseline_valid,
        "Required standard-JEPA and unconstrained-multihead baselines share the declared capacity contract.",
        families=sorted(str(name) for name in by_family),
        full_model_trainable_parameters=reference_match.get("full_model_trainable_parameters"),
        predictor_flops=reference_match.get("predictor_flops"),
        parameter_tolerance=reference_match.get("parameter_tolerance"),
        predictor_flops_tolerance=reference_match.get("predictor_flops_tolerance"),
        shared_capacity_contract=(
            baseline_capacity_matches(standard_match)
            and baseline_capacity_matches(multihead_match)
        ),
    )

    usage = recipe.get("usage", {})
    experiments = recipe.get("experiments", [])
    usage_mode = usage.get("mode")
    supported_usage = {"terminal_readout", "repeated_transition", "factor_analysis"}
    rollout_protocols = [
        experiment
        for experiment in experiments
        if experiment.get("kind") == "protocol_only"
    ]
    usage_valid = usage_mode in supported_usage
    if usage_mode == "repeated_transition":
        usage_valid = usage_valid and bool(usage.get("rollout_horizons")) and any(
            experiment.get("horizons") == usage.get("rollout_horizons")
            and any(
                metric.get("id") == "normalized_mse_by_horizon"
                and metric.get("direction") == "lower_is_better"
                for metric in experiment.get("metrics", [])
                if isinstance(metric, Mapping)
            )
            for experiment in rollout_protocols
        )
    audit.add(
        "usage_mode_coverage",
        usage_valid,
        "The selected use mode has a matching evaluation protocol.",
        mode=usage_mode,
        rollout_horizons=usage.get("rollout_horizons"),
    )

    claims = recipe.get("claims", {})
    claim_items = claims.get("items", [])
    experiment_by_id = {
        experiment.get("id"): experiment
        for experiment in experiments
        if isinstance(experiment, Mapping) and isinstance(experiment.get("id"), str)
    }
    baseline_ids = {
        baseline.get("id")
        for baseline in baselines
        if isinstance(baseline, Mapping) and isinstance(baseline.get("id"), str)
    }
    required_claim_fields = {
        "id",
        "status",
        "model_id",
        "metric_id",
        "direction",
        "split",
        "horizons",
        "baseline_ids",
        "audit_ids",
        "experiment_ids",
        "usage_mode",
    }

    def structured_claim_is_covered(item: Mapping[str, Any]) -> bool:
        evidence_ids = item.get("experiment_ids", [])
        if (
            set(item) != required_claim_fields
            or item.get("status") != "planned"
            or item.get("model_id") != "opf"
            or item.get("direction")
            not in {"lower_is_better", "higher_is_better"}
            or item.get("usage_mode") != usage_mode
            or not isinstance(evidence_ids, list)
            or not evidence_ids
            or not set(evidence_ids).issubset(experiment_by_id)
            or not set(item.get("baseline_ids", [])).issubset(baseline_ids)
        ):
            return False
        for experiment_id in evidence_ids:
            experiment = experiment_by_id[experiment_id]
            metrics = experiment.get("metrics", [])
            metric_covered = any(
                isinstance(metric, Mapping)
                and metric.get("id") == item.get("metric_id")
                and metric.get("direction") == item.get("direction")
                for metric in metrics
            )
            if not (
                metric_covered
                and experiment.get("split") == item.get("split")
                and experiment.get("horizons") == item.get("horizons")
                and set(item.get("baseline_ids", [])).issubset(
                    experiment.get("baselines", [])
                )
                and set(item.get("audit_ids", [])).issubset(
                    experiment.get("audits", [])
                )
                and item.get("id") in experiment.get("tests_claims", [])
            ):
                return False
        return True

    claims_covered = (
        isinstance(claim_items, list)
        and bool(claim_items)
        and all(
            isinstance(item, Mapping) and structured_claim_is_covered(item)
            for item in claim_items
        )
    )
    claims_policy_valid = (
        claims.get("scientific_results_claimed") is False
        and claims_covered
        and isinstance(claims.get("prohibited_inferences"), list)
        and bool(claims.get("prohibited_inferences"))
    )
    audit.add(
        "claim_coverage",
        claims_policy_valid,
        "All claims require named evidence; this smoke fixture declares no scientific result claims.",
        claim_count=(len(claim_items) if isinstance(claim_items, list) else None),
        experiment_ids=sorted(experiment_by_id),
        scientific_results_claimed=claims.get("scientific_results_claimed"),
    )

    design_task = design.get("task", {})
    design_observation = design.get("observation", {})
    design_context = design_observation.get("context", {})
    design_target = design_observation.get("target", {})
    design_model = design.get("model", {})
    design_state = design_model.get("state", {})
    design_adapter = design_observation.get("adapter", {})
    design_capacity = design_model.get("capacity", {})
    design_losses = design.get("losses", {})
    design_baselines = design.get("baselines", [])
    design_families = {
        value.get("kind", value.get("family"))
        for value in design_baselines
        if isinstance(value, dict)
    }
    source_protocols = {
        value.get("id"): value
        for value in experiments
        if isinstance(value, Mapping) and value.get("kind") == "protocol_only"
    }
    design_experiments = {
        value.get("id"): value
        for value in design.get("experiments", [])
        if isinstance(value, Mapping)
    }
    protocol_fields = {
        "id",
        "split",
        "horizons",
        "metrics",
        "baselines",
        "audits",
        "tests_claims",
    }
    protocols_consistent = all(
        experiment_id in design_experiments
        and all(
            design_experiments[experiment_id].get(field) == source.get(field)
            for field in protocol_fields
        )
        for experiment_id, source in source_protocols.items()
    )
    compiled_consistency = (
        design.get("schema_version") == "2.0"
        and design_task.get("system_id") == task.get("system_id")
        and design_task.get("identity_keys") == expected_identity
        and design_context.get("identity_keys") == expected_identity
        and design_target.get("identity_keys") == expected_identity
        and design_adapter.get("context_input_fields")
        == adapter.get("context_input_fields")
        and design_adapter.get("target_input_fields")
        == adapter.get("target_input_fields")
        and design_state.get("d") == d
        and design_state.get("K") == k
        and design_state.get("r") == r
        and design_state.get("coordinate_layout")
        == representation.get("coordinate_layout")
        and design_state.get("analysis") == representation.get("analysis")
        and design_state.get("synthesis") == representation.get("synthesis")
        and design_capacity == reference_match
        and design_losses.get("factor_prediction", {}).get("kind") == "factor_mse"
        and design_losses.get("projector_orthogonality", {}).get("kind")
        == "projector_gram"
        and design_losses.get("factor_activity", {}).get("kind")
        == "coordinate_std_floor"
        and design_losses.get("online_encoder_activity", {}).get("kind")
        == "coordinate_std_floor"
        and {"standard_jepa", "unconstrained_multihead"}.issubset(design_families)
        and design.get("usage", {}).get("mode") == usage_mode
        and protocols_consistent
        and design.get("claims") == claim_items
    )
    audit.add(
        "compiled_design_consistency",
        compiled_consistency,
        "Expected compiled design preserves source system, state, baseline, and usage decisions.",
        design_schema_version=design.get("schema_version"),
        design_system_id=design_task.get("system_id"),
        design_identity_keys=design_task.get("identity_keys"),
        design_state={
            key: design_state.get(key)
            for key in ("d", "K", "r", "coordinate_layout", "analysis", "synthesis")
        },
    )

    return audit


def build_report(
    recipe_path: Path,
    design_path: Path,
    recipe: Mapping[str, Any],
    audit: Audit,
) -> Dict[str, Any]:
    failed = [item["id"] for item in audit.checks if not item["passed"]]
    return {
        "schema_version": "jepa-anything.smoke-audit-report/v1",
        "recipe_id": recipe.get("recipe_id"),
        "status": "pass" if audit.passed else "fail",
        "checks_passed": sum(1 for item in audit.checks if item["passed"]),
        "checks_total": len(audit.checks),
        "failed_check_ids": failed,
        "inputs": {
            "recipe": {
                "path": recipe_path.as_posix(),
                "sha256": sha256_file(recipe_path),
            },
            "compiled_design": {
                "path": design_path.as_posix(),
                "sha256": sha256_file(design_path),
            },
        },
        "checks": audit.checks,
        "evidence_scope": {
            "training_executed": False,
            "checkpoint_produced": False,
            "scientific_result": False,
            "learned_factor_activity_measured": False,
            "allowed_interpretation": (
                "Structural smoke audit of a synthetic fixture; activity values "
                "come from oracle state coordinates, not a learned model."
            ),
            "not_supported": recipe.get("claims", {}).get("prohibited_inferences", []),
        },
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. Parent directories are created as needed.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout. Exit status still reports pass (0) or fail (1).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        recipe_path = args.recipe.resolve()
        design_path = args.design.resolve()
        recipe = load_json(recipe_path)
        design = load_json(design_path)
        audit = audit_recipe(recipe, design)
        report = build_report(recipe_path, design_path, recipe, audit)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"recipe audit could not run: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered, end="")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
