#!/usr/bin/env python3
"""Deterministically validate a JEPA Anything task-design configuration.

JSON support uses only the Python standard library. YAML is accepted when
PyYAML is installed. This validates a design contract; it does not run data,
instantiate a model, or establish empirical claims.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "2.0"
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "task",
    "observation",
    "model",
    "losses",
    "usage",
    "baselines",
    "audits",
    "experiments",
    "claims",
    "outputs",
}
USAGE_MODES = {"terminal_readout", "repeated_transition", "factor_analysis"}
ORTHOGONALITY_MODES = {"soft_gram", "qr_init", "qr_retraction"}
AVAILABILITY = {"observed_by_context_end", "known_at_prediction_time"}
REQUIRED_BASELINE_KINDS = {"standard_jepa", "unconstrained_multihead"}
REQUIRED_GEOMETRY_CHECKS = {
    "state_shape",
    "coordinate_concatenation",
    "within_projector_orthonormality",
    "cross_projector_orthogonality",
    "analysis_full_rank",
    "analysis_condition_number",
    "pseudoinverse_state_synthesis",
}
REQUIRED_LEAKAGE_CHECKS = {
    "system_identity",
    "temporal_separation",
    "feature_lineage",
    "descriptor_availability",
    "adapter_lineage",
    "conditioning_lineage",
}
REQUIRED_CLAIM_AUDITS = {
    "geometry",
    "factor_activity",
    "online_encoder_activity",
    "target_leakage",
    "capacity_matching",
}
METRIC_DIRECTIONS = {"lower_is_better", "higher_is_better"}
DESCRIPTOR_SOURCE_TYPES = {"context_source", "prediction_request", "static_metadata"}
EXOGENOUS_SOURCE_TYPES = {"context_source", "external_plan", "static_metadata"}
SEMANTIC_ATTRIBUTION = re.compile(
    r"(?:\bpc_\d{3}\b\s+(?:is|represents?|encodes?|means?|corresponds?\s+to)\b"
    r"|\bpc_\d{3}\b\s*(?:是|代表|表示|编码|对应(?:于)?))",
    re.IGNORECASE,
)
TASK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
COORDINATE_ID_PATTERN = re.compile(r"^pc_\d{3}$", re.IGNORECASE)


class ConfigLoadError(RuntimeError):
    """Raised when a configuration cannot be read or parsed."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and value > 0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


class DesignValidator:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.checks: list[dict[str, str]] = []

    @property
    def error_count(self) -> int:
        return sum(check["status"] == "fail" for check in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(check["status"] == "warning" for check in self.checks)

    def add(self, code: str, status: str, path: str, message: str) -> None:
        self.checks.append(
            {"code": code, "status": status, "path": path, "message": message}
        )

    def fail(self, code: str, path: str, message: str) -> None:
        self.add(code, "fail", path, message)

    def passed(self, code: str, path: str, message: str) -> None:
        self.add(code, "pass", path, message)

    def group_pass(self, start_errors: int, code: str, path: str, message: str) -> None:
        if self.error_count == start_errors:
            self.passed(code, path, message)

    def require_mapping(self, value: Any, path: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            self.fail("FIELD_INVALID", path, "Expected an object.")
            return {}
        return value

    def require_list(self, value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
        if not isinstance(value, list):
            self.fail("FIELD_INVALID", path, "Expected a list.")
            return []
        if nonempty and not value:
            self.fail("FIELD_INVALID", path, "Expected a non-empty list.")
        return value

    def require_string(self, value: Any, path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            self.fail("FIELD_INVALID", path, "Expected a non-empty string.")
            return ""
        return value

    def require_string_list(
        self,
        value: Any,
        path: str,
        *,
        nonempty: bool = False,
        unique: bool = True,
    ) -> list[str]:
        raw = self.require_list(value, path, nonempty=nonempty)
        result: list[str] = []
        valid = True
        for index, item in enumerate(raw):
            if not isinstance(item, str) or not item.strip():
                self.fail("FIELD_INVALID", f"{path}[{index}]", "Expected a non-empty string.")
                valid = False
            else:
                result.append(item)
        if unique and valid and len(result) != len(set(result)):
            self.fail("FIELD_INVALID", path, "List entries must be unique.")
        return result

    def validate(self) -> dict[str, Any]:
        if not isinstance(self.config, Mapping):
            self.fail("SCHEMA_REQUIRED", "$", "The root value must be an object.")
            return self.report()

        self.validate_schema()
        self.validate_coordinate_naming()
        self.validate_system_identity()
        self.validate_leakage()
        self.validate_state_and_losses()
        self.validate_usage()
        self.validate_baselines()
        self.validate_audits()
        self.validate_claim_coverage()
        self.validate_outputs()
        return self.report()

    def validate_coordinate_naming(self) -> None:
        """Reject direct semantic attribution to canonical prediction coordinates."""

        start = self.error_count

        def visit(value: Any, path: str) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if isinstance(key, str) and COORDINATE_ID_PATTERN.fullmatch(key):
                        self.fail(
                            "SEMANTIC_FACTOR_NAMING_FORBIDDEN",
                            f"{path}.{key}",
                            "Semantic coordinate-name mappings are forbidden at design time.",
                        )
                    visit(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")
            elif isinstance(value, str) and SEMANTIC_ATTRIBUTION.search(value):
                self.fail(
                    "SEMANTIC_FACTOR_NAMING_FORBIDDEN",
                    path,
                    "Prediction coordinates may not be assigned domain semantics at design time.",
                )

        visit(self.config, "$")
        self.group_pass(
            start,
            "COORDINATE_NAMING_VALID",
            "$",
            "Canonical prediction coordinates have no asserted domain semantics.",
        )

    def validate_schema(self) -> None:
        start = self.error_count
        keys = set(self.config)
        for key in sorted(REQUIRED_TOP_LEVEL - keys):
            self.fail("SCHEMA_REQUIRED", f"$.{key}", f"Missing required top-level key: {key}.")
        for key in sorted(keys - REQUIRED_TOP_LEVEL):
            self.fail("SCHEMA_UNKNOWN_KEY", f"$.{key}", f"Unknown top-level key: {key}.")
        if self.config.get("schema_version") != SCHEMA_VERSION:
            self.fail(
                "SCHEMA_VERSION_UNSUPPORTED",
                "$.schema_version",
                f"Expected schema_version {SCHEMA_VERSION!r}.",
            )
        for key in ("task", "observation", "model", "losses", "usage", "audits", "outputs"):
            if key in self.config:
                self.require_mapping(self.config[key], f"$.{key}")
        for key in ("baselines", "experiments", "claims"):
            if key in self.config:
                self.require_list(self.config[key], f"$.{key}", nonempty=True)
        task = _as_mapping(self.config.get("task"))
        task_id = self.require_string(task.get("id"), "$.task.id")
        if task_id and not TASK_ID_PATTERN.fullmatch(task_id):
            self.fail(
                "FIELD_INVALID",
                "$.task.id",
                "Task ID must use lowercase letters, digits, hyphens, or underscores.",
            )
        self.require_string(task.get("summary"), "$.task.summary")
        if "provenance" in task:
            self.require_mapping(task.get("provenance"), "$.task.provenance")
        if "metadata" in task:
            self.require_mapping(task.get("metadata"), "$.task.metadata")
        self.group_pass(start, "SCHEMA_VALID", "$", "Schema version and top-level structure are valid.")

    def validate_system_identity(self) -> None:
        start = self.error_count
        task = _as_mapping(self.config.get("task"))
        observation = _as_mapping(self.config.get("observation"))
        context = _as_mapping(observation.get("context"))
        target = _as_mapping(observation.get("target"))
        expected = self.require_string(task.get("system_id"), "$.task.system_id")
        context_id = self.require_string(context.get("system_id"), "$.observation.context.system_id")
        target_id = self.require_string(target.get("system_id"), "$.observation.target.system_id")
        expected_identity = self.require_string_list(
            task.get("identity_keys"), "$.task.identity_keys", nonempty=True
        )
        context_identity = self.require_string_list(
            context.get("identity_keys"),
            "$.observation.context.identity_keys",
            nonempty=True,
        )
        target_identity = self.require_string_list(
            target.get("identity_keys"),
            "$.observation.target.identity_keys",
            nonempty=True,
        )
        if (
            not expected
            or context_id != expected
            or target_id != expected
            or not expected_identity
            or context_identity != expected_identity
            or target_identity != expected_identity
        ):
            self.fail(
                "SYSTEM_IDENTITY_MISMATCH",
                "$.observation",
                "Context and target must share task.system_id and exactly task.identity_keys.",
            )
        self.group_pass(
            start,
            "SYSTEM_IDENTITY_VALID",
            "$.observation",
            "Context and target declare the same underlying system and instance identity keys.",
        )

    def _validate_available_fields(
        self,
        container: Any,
        path: str,
        *,
        with_role: bool,
        allowed_source_types: set[str],
        source_fields: Mapping[str, set[str]],
        source_cutoffs: Mapping[str, int],
        context_end: Any,
        target_fields: set[str],
    ) -> None:
        obj = self.require_mapping(container, path)
        fields = self.require_list(obj.get("fields"), f"{path}.fields")
        names: set[str] = set()
        for index, raw in enumerate(fields):
            item_path = f"{path}.fields[{index}]"
            item = self.require_mapping(raw, item_path)
            name = self.require_string(item.get("name"), f"{item_path}.name")
            if name in names:
                self.fail("FIELD_INVALID", f"{item_path}.name", "Conditioning field names must be unique.")
            names.add(name)
            availability = item.get("availability")
            if availability not in AVAILABILITY:
                self.fail(
                    "LEAKAGE_INPUT_UNAVAILABLE",
                    f"{item_path}.availability",
                    "Input must be observed by context end or known at prediction time.",
                )
            if with_role and item.get("role") not in {"index_only", "conditioning"}:
                self.fail("FIELD_INVALID", f"{item_path}.role", "Descriptor role is invalid.")

            lineage = self.require_mapping(item.get("lineage"), f"{item_path}.lineage")
            source_id = self.require_string(lineage.get("source_id"), f"{item_path}.lineage.source_id")
            source_type = lineage.get("source_type")
            if source_type not in allowed_source_types:
                self.fail(
                    "LEAKAGE_INPUT_UNAVAILABLE",
                    f"{item_path}.lineage.source_type",
                    f"Expected one of {sorted(allowed_source_types)}.",
                )
            available_through = lineage.get("available_through")
            if not _is_int(available_through):
                self.fail(
                    "FIELD_INVALID",
                    f"{item_path}.lineage.available_through",
                    "Expected an integer offset.",
                )
            elif _is_int(context_end) and available_through > context_end:
                self.fail(
                    "LEAKAGE_INPUT_UNAVAILABLE",
                    f"{item_path}.lineage.available_through",
                    "Conditioning input is unavailable at prediction time.",
                )
            if source_type == "context_source":
                if source_id not in source_fields or name not in source_fields.get(source_id, set()):
                    self.fail(
                        "LEAKAGE_SOURCE_UNKNOWN",
                        f"{item_path}.lineage",
                        "Context-source lineage must name a declared source containing this field.",
                    )
                elif (
                    _is_int(available_through)
                    and source_id in source_cutoffs
                    and available_through > source_cutoffs[source_id]
                ):
                    self.fail(
                        "LEAKAGE_SOURCE_UNAVAILABLE",
                        f"{item_path}.lineage.available_through",
                        "Lineage cannot claim availability beyond its context source cutoff.",
                    )
                if availability != "observed_by_context_end":
                    self.fail(
                        "LEAKAGE_INPUT_UNAVAILABLE",
                        f"{item_path}.availability",
                        "A context-source field must be observed by context end.",
                    )
            elif source_type in {"prediction_request", "external_plan"}:
                if availability != "known_at_prediction_time":
                    self.fail(
                        "LEAKAGE_INPUT_UNAVAILABLE",
                        f"{item_path}.availability",
                        "Prediction requests and external plans must be known at prediction time.",
                    )
            if name and name in target_fields:
                self.fail(
                    "LEAKAGE_TARGET_DERIVED_CONTEXT",
                    f"{item_path}.name",
                    "Conditioning fields must not reuse a predicted target field name.",
                )

    def validate_leakage(self) -> None:
        start = self.error_count
        observation = _as_mapping(self.config.get("observation"))
        adapter = self.require_mapping(observation.get("adapter"), "$.observation.adapter")
        self.require_string(adapter.get("kind"), "$.observation.adapter.kind")
        context_input_fields = self.require_string_list(
            adapter.get("context_input_fields"),
            "$.observation.adapter.context_input_fields",
            nonempty=True,
        )
        target_input_fields = self.require_string_list(
            adapter.get("target_input_fields"),
            "$.observation.adapter.target_input_fields",
            nonempty=True,
        )
        self.require_string(adapter.get("tokenization"), "$.observation.adapter.tokenization")
        if not _is_positive_int(adapter.get("output_dim")):
            self.fail("FIELD_INVALID", "$.observation.adapter.output_dim", "Expected a positive integer.")

        context = self.require_mapping(observation.get("context"), "$.observation.context")
        target = self.require_mapping(observation.get("target"), "$.observation.target")
        context_window = self.require_mapping(context.get("window"), "$.observation.context.window")
        target_window = self.require_mapping(target.get("window"), "$.observation.target.window")
        cs, ce = context_window.get("start"), context_window.get("end")
        ts, te = target_window.get("start"), target_window.get("end")
        if not all(_is_int(value) for value in (cs, ce, ts, te)):
            self.fail("FIELD_INVALID", "$.observation.*.window", "Window bounds must be integers.")
        elif not (cs <= ce < ts <= te):
            self.fail(
                "LEAKAGE_TEMPORAL_OVERLAP",
                "$.observation.*.window",
                "Require context.start <= context.end < target.start <= target.end.",
            )

        target_derived = context.get("target_derived_fields")
        if not isinstance(target_derived, list):
            self.fail("FIELD_INVALID", "$.observation.context.target_derived_fields", "Expected a list.")
        elif target_derived:
            self.fail(
                "LEAKAGE_TARGET_DERIVED_CONTEXT",
                "$.observation.context.target_derived_fields",
                "Context cannot contain target-derived fields.",
            )

        sources = self.require_list(context.get("sources"), "$.observation.context.sources", nonempty=True)
        source_ids: set[str] = set()
        source_fields: dict[str, set[str]] = {}
        source_cutoffs: dict[str, int] = {}
        for index, raw in enumerate(sources):
            item_path = f"$.observation.context.sources[{index}]"
            source = self.require_mapping(raw, item_path)
            source_id = self.require_string(source.get("id"), f"{item_path}.id")
            if source_id in source_ids:
                self.fail("FIELD_INVALID", f"{item_path}.id", "Context source IDs must be unique.")
            source_ids.add(source_id)
            fields = self.require_string_list(source.get("fields"), f"{item_path}.fields", nonempty=True)
            source_fields[source_id] = set(fields)
            available = source.get("available_through")
            if not _is_int(available):
                self.fail("FIELD_INVALID", f"{item_path}.available_through", "Expected an integer offset.")
            else:
                source_cutoffs[source_id] = available
                if _is_int(ce) and available > ce:
                    self.fail(
                        "LEAKAGE_SOURCE_UNAVAILABLE",
                        f"{item_path}.available_through",
                        "Context source contains values after context end.",
                    )

        target_field_list = self.require_string_list(
            target.get("fields"), "$.observation.target.fields", nonempty=True
        )
        target_fields = set(target_field_list)
        available_context_fields = set().union(*source_fields.values()) if source_fields else set()
        missing_context_lineage = set(context_input_fields) - available_context_fields
        if missing_context_lineage:
            self.fail(
                "LEAKAGE_SOURCE_UNKNOWN",
                "$.observation.adapter.context_input_fields",
                f"Context adapter fields lack declared source lineage: {sorted(missing_context_lineage)}.",
            )
        if target_input_fields != target_field_list:
            self.fail(
                "LEAKAGE_SOURCE_UNKNOWN",
                "$.observation.adapter.target_input_fields",
                "Target adapter fields must exactly match observation.target.fields in order.",
            )
        target_sources = self.require_string_list(
            target.get("source_ids"), "$.observation.target.source_ids", nonempty=True
        )
        unknown_target_sources = set(target_sources) - source_ids
        if unknown_target_sources:
            self.fail(
                "LEAKAGE_SOURCE_UNKNOWN",
                "$.observation.target.source_ids",
                f"Target references undeclared source IDs: {sorted(unknown_target_sources)}.",
            )
        target_source_fields = set().union(
            *(source_fields.get(source_id, set()) for source_id in target_sources)
        ) if target_sources else set()
        unbound_target_fields = target_fields - target_source_fields
        if unbound_target_fields:
            self.fail(
                "LEAKAGE_SOURCE_UNKNOWN",
                "$.observation.target.fields",
                "Target fields lack declared same-system source lineage: "
                f"{sorted(unbound_target_fields)}.",
            )

        self._validate_available_fields(
            observation.get("target_descriptor"),
            "$.observation.target_descriptor",
            with_role=True,
            allowed_source_types=DESCRIPTOR_SOURCE_TYPES,
            source_fields=source_fields,
            source_cutoffs=source_cutoffs,
            context_end=ce,
            target_fields=target_fields,
        )
        self._validate_available_fields(
            observation.get("exogenous_inputs"),
            "$.observation.exogenous_inputs",
            with_role=False,
            allowed_source_types=EXOGENOUS_SOURCE_TYPES,
            source_fields=source_fields,
            source_cutoffs=source_cutoffs,
            context_end=ce,
            target_fields=target_fields,
        )
        self.group_pass(
            start,
            "TARGET_LEAKAGE_CONTROLS_VALID",
            "$.observation",
            "Temporal separation, declared lineage, and input availability controls are valid.",
        )

    def _validate_coordinate_list(self, value: Any, expected: set[str], path: str) -> bool:
        actual = _string_set(value)
        ok = isinstance(value, list) and len(value) == len(actual) and actual == expected
        if not ok:
            self.fail(
                "FACTOR_ACTIVITY_INCOMPLETE",
                path,
                "Coordinate list must contain every canonical coordinate exactly once.",
            )
        return ok

    def _validate_factor_dimension_coordinates(
        self,
        value: Any,
        expected: list[dict[str, int | str]],
        path: str,
    ) -> bool:
        raw = self.require_list(value, path, nonempty=True)
        normalized: list[dict[str, int | str]] = []
        valid = True
        for index, item in enumerate(raw):
            item_path = f"{path}[{index}]"
            coordinate = self.require_mapping(item, item_path)
            if set(coordinate) != {"factor_id", "coordinate_index"}:
                self.fail(
                    "FACTOR_ACTIVITY_INCOMPLETE",
                    item_path,
                    "Each coordinate must contain only factor_id and coordinate_index.",
                )
                valid = False
            factor_id = coordinate.get("factor_id")
            coordinate_index = coordinate.get("coordinate_index")
            if not isinstance(factor_id, str) or not _is_int(coordinate_index):
                self.fail(
                    "FACTOR_ACTIVITY_INCOMPLETE",
                    item_path,
                    "Invalid factor-dimension coordinate reference.",
                )
                valid = False
            else:
                normalized.append(
                    {"factor_id": factor_id, "coordinate_index": coordinate_index}
                )
        if valid and normalized != expected:
            self.fail(
                "FACTOR_ACTIVITY_INCOMPLETE",
                path,
                "Coordinate list must cover every factor dimension exactly once in canonical order.",
            )
            valid = False
        return valid

    def _validate_encoder_dimensions(self, value: Any, d: Any, path: str) -> bool:
        raw = self.require_list(value, path, nonempty=True)
        expected = list(range(d)) if _is_positive_int(d) else []
        valid = raw == expected
        if not valid:
            self.fail(
                "FACTOR_ACTIVITY_INCOMPLETE",
                path,
                "Online-encoder dimensions must be exactly 0..d-1.",
            )
        return valid

    def validate_state_and_losses(self) -> None:
        start_state = self.error_count
        model = self.require_mapping(self.config.get("model"), "$.model")
        self.require_string(model.get("id"), "$.model.id")
        state = self.require_mapping(model.get("state"), "$.model.state")
        d, k, r = state.get("d"), state.get("K"), state.get("r")
        if not all(_is_positive_int(value) for value in (d, k, r)):
            self.fail("FIELD_INVALID", "$.model.state", "d, K, and r must be positive integers.")
        elif k * r != d:
            self.fail(
                "STATE_DIMENSION_MISMATCH",
                "$.model.state",
                f"Expected K * r == d, got {k} * {r} != {d}.",
            )
        if state.get("coordinate_layout") != "concatenate":
            self.fail(
                "STATE_COMPOSITION_INVALID",
                "$.model.state.coordinate_layout",
                "Factor coordinates must use canonical concatenation order.",
            )

        analysis = self.require_mapping(state.get("analysis"), "$.model.state.analysis")
        expected_projector_shape = [d, r] if _is_positive_int(d) and _is_positive_int(r) else []
        if (
            analysis.get("kind") != "learned_projectors"
            or analysis.get("projector_count") != k
            or analysis.get("projector_shape") != expected_projector_shape
            or analysis.get("target_stop_gradient") is not True
        ):
            self.fail(
                "STATE_COMPOSITION_INVALID",
                "$.model.state.analysis",
                "Analysis requires K learned d-by-r projectors over a stop-gradient target.",
            )
        orthogonality_mode = analysis.get("orthogonality_mode", "soft_gram")
        qr_retraction = analysis.get("qr_retraction")
        if orthogonality_mode not in ORTHOGONALITY_MODES:
            self.fail(
                "ORTHOGONALITY_STRATEGY_INVALID",
                "$.model.state.analysis.orthogonality_mode",
                "Use soft_gram, qr_init, or qr_retraction.",
            )
        elif orthogonality_mode == "qr_retraction":
            retraction = self.require_mapping(
                qr_retraction,
                "$.model.state.analysis.qr_retraction",
            )
            if (
                retraction.get("timing") != "after_optimizer_step"
                or not _is_positive_int(retraction.get("frequency"))
                or retraction.get("sign_canonicalization") != "positive_r_diagonal"
                or retraction.get("training_compute_accounting") != "report_separately"
            ):
                self.fail(
                    "ORTHOGONALITY_STRATEGY_INVALID",
                    "$.model.state.analysis.qr_retraction",
                    "QR retraction requires after_optimizer_step timing, a positive "
                    "frequency, positive_r_diagonal sign canonicalization, and "
                    "separate training-compute accounting.",
                )
        elif qr_retraction is not None:
            self.fail(
                "ORTHOGONALITY_STRATEGY_INVALID",
                "$.model.state.analysis.qr_retraction",
                "qr_retraction settings are allowed only in qr_retraction mode.",
            )
        synthesis = self.require_mapping(state.get("synthesis"), "$.model.state.synthesis")
        if (
            synthesis.get("kind") != "moore_penrose_pseudoinverse"
            or synthesis.get("analysis_map") != "projector_transpose"
            or synthesis.get("require_full_rank") is not True
        ):
            self.fail(
                "STATE_COMPOSITION_INVALID",
                "$.model.state.synthesis",
                "Complete state synthesis must use the pseudoinverse of the projector-transpose analysis map and require full rank.",
            )

        factors = self.require_list(state.get("factors"), "$.model.state.factors", nonempty=True)
        expected_ids = {f"pc_{index:03d}" for index in range(k)} if _is_positive_int(k) else set()
        if _is_positive_int(k) and len(factors) != k:
            self.fail(
                "COORDINATE_CONTRACT_INVALID",
                "$.model.state.factors",
                f"Expected exactly K={k} coordinate objects.",
            )
        for index, raw in enumerate(factors):
            item_path = f"$.model.state.factors[{index}]"
            factor = self.require_mapping(raw, item_path)
            extra = set(factor) - {"id", "index", "dim"}
            if extra:
                self.fail(
                    "COORDINATE_CONTRACT_INVALID",
                    item_path,
                    "Factor objects may contain only id, index, and dim; semantic labels are forbidden.",
                )
            coordinate_id = factor.get("id")
            if coordinate_id != f"pc_{index:03d}" or factor.get("index") != index or factor.get("dim") != r:
                self.fail(
                    "COORDINATE_CONTRACT_INVALID",
                    item_path,
                    "Coordinate ID, index, or dimension does not match its canonical position.",
                )

        encoder = self.require_mapping(model.get("encoder"), "$.model.encoder")
        self.require_string(encoder.get("kind"), "$.model.encoder.kind")
        if encoder.get("shared_context_target") is not True:
            self.fail("FIELD_INVALID", "$.model.encoder.shared_context_target", "Expected true.")
        if _is_positive_int(d) and encoder.get("output_dim") != d:
            self.fail("STATE_DIMENSION_MISMATCH", "$.model.encoder.output_dim", "Encoder output_dim must equal d.")
        predictor = self.require_mapping(model.get("predictor"), "$.model.predictor")
        if predictor.get("kind") != "opf" or (_is_positive_int(d) and predictor.get("output_dim") != d):
            self.fail("STATE_DIMENSION_MISMATCH", "$.model.predictor", "OPF predictor output_dim must equal d.")
        capacity = self.require_mapping(model.get("capacity"), "$.model.capacity")
        if not _is_positive_int(capacity.get("full_model_trainable_parameters")):
            self.fail(
                "FIELD_INVALID",
                "$.model.capacity.full_model_trainable_parameters",
                "Expected a positive integer.",
            )
        if not _is_positive_int(capacity.get("predictor_flops")):
            self.fail("FIELD_INVALID", "$.model.capacity.predictor_flops", "Expected a positive integer.")
        if not _is_positive_int(capacity.get("training_steps")):
            self.fail("FIELD_INVALID", "$.model.capacity.training_steps", "Expected a positive integer.")
        for field in ("parameter_tolerance", "predictor_flops_tolerance"):
            tolerance = capacity.get(field)
            if not _is_number(tolerance) or not 0 <= tolerance < 1:
                self.fail("FIELD_INVALID", f"$.model.capacity.{field}", "Expected a number in [0, 1).")
        if capacity.get("parameter_scope") != "full_trainable_model":
            self.fail("FIELD_INVALID", "$.model.capacity.parameter_scope", "Use full_trainable_model.")
        if capacity.get("flops_scope") != "predictor_forward_per_target":
            self.fail("FIELD_INVALID", "$.model.capacity.flops_scope", "Use predictor_forward_per_target.")
        self.group_pass(
            start_state,
            "STATE_COMPOSITION_VALID",
            "$.model.state",
            "State dimensions, learned-projector analysis, coordinate concatenation, and pseudoinverse synthesis are valid.",
        )

        start_activity = self.error_count
        losses = self.require_mapping(self.config.get("losses"), "$.losses")
        expected_factor_coordinates = [
            {"factor_id": f"pc_{factor_index:03d}", "coordinate_index": coordinate_index}
            for factor_index in range(k if _is_positive_int(k) else 0)
            for coordinate_index in range(r if _is_positive_int(r) else 0)
        ]
        prediction = self.require_mapping(
            losses.get("factor_prediction"), "$.losses.factor_prediction"
        )
        if (
            prediction.get("enabled") is not True
            or prediction.get("kind") != "factor_mse"
            or not _is_positive_number(prediction.get("weight"))
        ):
            self.fail("FIELD_INVALID", "$.losses.factor_prediction", "Factor-MSE loss contract is invalid.")

        orthogonality = self.require_mapping(
            losses.get("projector_orthogonality"), "$.losses.projector_orthogonality"
        )
        components = self.require_string_list(
            orthogonality.get("components"),
            "$.losses.projector_orthogonality.components",
            nonempty=True,
        )
        if (
            orthogonality.get("enabled") is not True
            or orthogonality.get("kind") != "projector_gram"
            or not _is_positive_number(orthogonality.get("weight"))
            or set(components) != {"within_projector", "cross_projector"}
        ):
            self.fail(
                "FIELD_INVALID",
                "$.losses.projector_orthogonality",
                "Projector-Gram loss must include within- and cross-projector terms.",
            )
        self._validate_coordinate_list(
            orthogonality.get("projectors"),
            expected_ids,
            "$.losses.projector_orthogonality.projectors",
        )

        activity = self.require_mapping(
            losses.get("factor_activity"), "$.losses.factor_activity"
        )
        if (
            activity.get("enabled") is not True
            or activity.get("kind") != "coordinate_std_floor"
            or activity.get("scope") != "projected_target_factors"
            or not _is_positive_number(activity.get("weight"))
            or not _is_positive_number(activity.get("min_std"))
            or not _is_positive_number(activity.get("epsilon"))
        ):
            self.fail(
                "FACTOR_ACTIVITY_INCOMPLETE",
                "$.losses.factor_activity",
                "Every projected target coordinate requires a positive standard-deviation floor.",
            )
        self._validate_factor_dimension_coordinates(
            activity.get("coordinates"),
            expected_factor_coordinates,
            "$.losses.factor_activity.coordinates",
        )

        online = self.require_mapping(
            losses.get("online_encoder_activity"), "$.losses.online_encoder_activity"
        )
        if (
            online.get("enabled") is not True
            or online.get("kind") != "coordinate_std_floor"
            or online.get("scope") != "online_context_encoder"
            or not _is_positive_number(online.get("weight"))
            or not _is_positive_number(online.get("min_std"))
            or not _is_positive_number(online.get("epsilon"))
        ):
            self.fail(
                "FACTOR_ACTIVITY_INCOMPLETE",
                "$.losses.online_encoder_activity",
                "Every online context-encoder dimension requires a positive standard-deviation floor.",
            )
        self._validate_encoder_dimensions(
            online.get("dimensions"), d, "$.losses.online_encoder_activity.dimensions"
        )

        audits = _as_mapping(self.config.get("audits"))
        factor_audit = self.require_mapping(audits.get("factor_activity"), "$.audits.factor_activity")
        if factor_audit.get("enabled") is not True or factor_audit.get("metric") != "coordinate_std" or not _is_positive_number(factor_audit.get("minimum")):
            self.fail("FACTOR_ACTIVITY_INCOMPLETE", "$.audits.factor_activity", "Factor-activity audit is invalid.")
        self._validate_factor_dimension_coordinates(
            factor_audit.get("coordinates"),
            expected_factor_coordinates,
            "$.audits.factor_activity.coordinates",
        )
        online_audit = self.require_mapping(
            audits.get("online_encoder_activity"), "$.audits.online_encoder_activity"
        )
        if (
            online_audit.get("enabled") is not True
            or online_audit.get("metric") != "coordinate_std"
            or not _is_positive_number(online_audit.get("minimum"))
        ):
            self.fail(
                "FACTOR_ACTIVITY_INCOMPLETE",
                "$.audits.online_encoder_activity",
                "Online-encoder activity audit is invalid.",
            )
        self._validate_encoder_dimensions(
            online_audit.get("dimensions"), d, "$.audits.online_encoder_activity.dimensions"
        )
        self.group_pass(
            start_activity,
            "FACTOR_ACTIVITY_CONFIGURED",
            "$.losses",
            "Every projected target coordinate and online encoder dimension has a standard-deviation floor and audit coverage.",
        )

    def validate_usage(self) -> None:
        start = self.error_count
        usage = self.require_mapping(self.config.get("usage"), "$.usage")
        mode = usage.get("mode")
        if mode not in USAGE_MODES:
            self.fail("USAGE_MODE_INVALID", "$.usage.mode", "Unknown use mode.")
            return
        mode_key = {
            "terminal_readout": "readout",
            "repeated_transition": "transition",
            "factor_analysis": "analysis",
        }[mode]
        forbidden = {"readout", "transition", "analysis"} - {mode_key}
        for key in forbidden:
            if key in usage:
                self.fail("USAGE_MODE_CONTRACT_INVALID", f"$.usage.{key}", f"{key} is not valid for mode {mode}.")
        details = self.require_mapping(usage.get(mode_key), f"$.usage.{mode_key}")
        if mode == "terminal_readout":
            self.require_string_list(details.get("tasks"), "$.usage.readout.tasks", nonempty=True)
            self.require_string_list(details.get("metrics"), "$.usage.readout.metrics", nonempty=True)
        elif mode == "repeated_transition":
            horizons = self.require_list(details.get("rollout_horizons"), "$.usage.transition.rollout_horizons", nonempty=True)
            if (
                any(not _is_positive_int(horizon) for horizon in horizons)
                or len(horizons) != len(set(horizons))
                or horizons != sorted(horizons)
            ):
                self.fail(
                    "USAGE_MODE_CONTRACT_INVALID",
                    "$.usage.transition.rollout_horizons",
                    "Horizons must be unique positive integers in ascending order.",
                )
            if details.get("condition_on_exogenous") not in {True, False}:
                self.fail("USAGE_MODE_CONTRACT_INVALID", "$.usage.transition.condition_on_exogenous", "Expected a boolean.")
        else:
            self.require_string_list(details.get("probes"), "$.usage.analysis.probes", nonempty=True)
            self.require_string_list(details.get("interventions"), "$.usage.analysis.interventions", nonempty=True)
            if details.get("coordinate_reporting") != "prediction_coordinates_only":
                self.fail("USAGE_MODE_CONTRACT_INVALID", "$.usage.analysis.coordinate_reporting", "Use prediction_coordinates_only.")
        self.group_pass(start, "USAGE_MODE_VALID", "$.usage", f"Use mode {mode} has a valid contract.")

    def validate_baselines(self) -> None:
        start = self.error_count
        baselines = self.require_list(self.config.get("baselines"), "$.baselines", nonempty=True)
        model = _as_mapping(self.config.get("model"))
        state = _as_mapping(model.get("state"))
        capacity = _as_mapping(model.get("capacity"))
        d, k, r = state.get("d"), state.get("K"), state.get("r")
        reference_id = model.get("id")
        reference_parameters = capacity.get("full_model_trainable_parameters")
        reference_flops = capacity.get("predictor_flops")
        reference_steps = capacity.get("training_steps")
        parameter_tolerance = capacity.get("parameter_tolerance")
        flops_tolerance = capacity.get("predictor_flops_tolerance")
        by_kind: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
        ids: set[str] = set()
        for index, raw in enumerate(baselines):
            path = f"$.baselines[{index}]"
            baseline = self.require_mapping(raw, path)
            baseline_id = self.require_string(baseline.get("id"), f"{path}.id")
            if baseline_id in ids:
                self.fail("FIELD_INVALID", f"{path}.id", "Baseline IDs must be unique.")
            if baseline_id and baseline_id == reference_id:
                self.fail(
                    "BASELINE_CAPACITY_MISMATCH",
                    f"{path}.id",
                    "A baseline ID must be distinct from the reference model ID.",
                )
            ids.add(baseline_id)
            kind = baseline.get("kind")
            if isinstance(kind, str):
                by_kind.setdefault(kind, []).append((index, baseline))

        for kind in sorted(REQUIRED_BASELINE_KINDS):
            entries = by_kind.get(kind, [])
            if len(entries) != 1:
                self.fail(
                    "BASELINE_REQUIRED_MISSING",
                    "$.baselines",
                    f"Expected exactly one baseline with kind {kind!r}.",
                )
                continue
            index, baseline = entries[0]
            path = f"$.baselines[{index}]"
            predictor = self.require_mapping(baseline.get("predictor"), f"{path}.predictor")
            if _is_positive_int(d) and predictor.get("output_dim") != d:
                self.fail("BASELINE_CAPACITY_MISMATCH", f"{path}.predictor.output_dim", "Baseline output_dim must equal d.")
            match = self.require_mapping(baseline.get("capacity_match"), f"{path}.capacity_match")
            if match.get("reference_model_id") != reference_id:
                self.fail("BASELINE_CAPACITY_MISMATCH", f"{path}.capacity_match.reference_model_id", "Wrong reference model.")
            params = match.get("full_model_trainable_parameters")
            if (
                _is_positive_int(reference_parameters)
                and _is_positive_int(params)
                and _is_number(parameter_tolerance)
            ):
                relative = abs(params - reference_parameters) / reference_parameters
                if relative > parameter_tolerance:
                    self.fail(
                        "BASELINE_CAPACITY_MISMATCH",
                        f"{path}.capacity_match.full_model_trainable_parameters",
                        "Full-model parameter budget exceeds matching tolerance.",
                    )
            else:
                self.fail(
                    "BASELINE_CAPACITY_MISMATCH",
                    f"{path}.capacity_match.full_model_trainable_parameters",
                    "Invalid full-model parameter budget.",
                )
            predictor_flops = match.get("predictor_flops")
            if (
                _is_positive_int(reference_flops)
                and _is_positive_int(predictor_flops)
                and _is_number(flops_tolerance)
            ):
                relative_flops = abs(predictor_flops - reference_flops) / reference_flops
                if relative_flops > flops_tolerance:
                    self.fail(
                        "BASELINE_FLOPS_MISMATCH",
                        f"{path}.capacity_match.predictor_flops",
                        "Predictor FLOPs exceed matching tolerance.",
                    )
            else:
                self.fail(
                    "BASELINE_FLOPS_MISMATCH",
                    f"{path}.capacity_match.predictor_flops",
                    "Invalid predictor FLOPs budget.",
                )
            if match.get("training_steps") != reference_steps:
                self.fail("BASELINE_CAPACITY_MISMATCH", f"{path}.capacity_match.training_steps", "Training steps must match exactly.")
            if match.get("parameter_tolerance") != parameter_tolerance:
                self.fail(
                    "BASELINE_CAPACITY_MISMATCH",
                    f"{path}.capacity_match.parameter_tolerance",
                    "The full-model parameter tolerance must be declared and match the reference contract.",
                )
            if match.get("predictor_flops_tolerance") != flops_tolerance:
                self.fail(
                    "BASELINE_FLOPS_MISMATCH",
                    f"{path}.capacity_match.predictor_flops_tolerance",
                    "The predictor-FLOPs tolerance must be declared and match the reference contract.",
                )
            if (
                match.get("parameter_scope") != capacity.get("parameter_scope")
                or match.get("flops_scope") != capacity.get("flops_scope")
            ):
                self.fail(
                    "BASELINE_CAPACITY_MISMATCH",
                    f"{path}.capacity_match",
                    "Parameter and FLOPs counting scopes must match the reference.",
                )
            if match.get("encoder_policy") != "same_family_and_width" or match.get("data_policy") != "same_samples_and_augmentations":
                self.fail("BASELINE_CAPACITY_MISMATCH", f"{path}.capacity_match", "Encoder and data policies must match the reference.")
            if kind == "unconstrained_multihead":
                if baseline.get("heads") != k or baseline.get("head_dim") != r or baseline.get("constraints") != []:
                    self.fail(
                        "BASELINE_UNCONSTRAINED_INVALID",
                        path,
                        "Unconstrained multihead baseline must use K heads of width r and no constraints.",
                    )
        self.group_pass(
            start,
            "BASELINES_CAPACITY_MATCHED",
            "$.baselines",
            "Required standard JEPA and unconstrained multihead baselines match declared full-model parameters, predictor FLOPs, and training budgets.",
        )

    def validate_audits(self) -> None:
        audits = self.require_mapping(self.config.get("audits"), "$.audits")
        geometry = self.require_mapping(audits.get("geometry"), "$.audits.geometry")
        geometry_checks = self.require_string_list(
            geometry.get("checks"), "$.audits.geometry.checks", nonempty=True
        )
        if geometry.get("enabled") is not True or not REQUIRED_GEOMETRY_CHECKS.issubset(set(geometry_checks)):
            self.fail("FIELD_INVALID", "$.audits.geometry", "Geometry audit is incomplete.")
        if geometry.get("full_rank_required") is not True:
            self.fail("FIELD_INVALID", "$.audits.geometry.full_rank_required", "Full-rank analysis-map audit is required.")
        if not _is_positive_number(geometry.get("max_condition_number")):
            self.fail("FIELD_INVALID", "$.audits.geometry.max_condition_number", "Expected a positive condition-number ceiling.")
        synthesis_tolerance = geometry.get("max_synthesis_nmse")
        if not _is_number(synthesis_tolerance) or synthesis_tolerance < 0:
            self.fail("FIELD_INVALID", "$.audits.geometry.max_synthesis_nmse", "Expected a non-negative synthesis-NMSE ceiling.")
        leakage = self.require_mapping(audits.get("target_leakage"), "$.audits.target_leakage")
        leakage_checks = self.require_string_list(
            leakage.get("checks"), "$.audits.target_leakage.checks", nonempty=True
        )
        if leakage.get("enabled") is not True or not REQUIRED_LEAKAGE_CHECKS.issubset(set(leakage_checks)):
            self.fail("FIELD_INVALID", "$.audits.target_leakage", "Target-leakage audit is incomplete.")
        matching = self.require_mapping(audits.get("capacity_matching"), "$.audits.capacity_matching")
        matching_baselines = self.require_string_list(
            matching.get("baseline_ids"), "$.audits.capacity_matching.baseline_ids", nonempty=True
        )
        baseline_ids = {
            baseline.get("id")
            for baseline in _as_list(self.config.get("baselines"))
            if isinstance(baseline, Mapping) and baseline.get("kind") in REQUIRED_BASELINE_KINDS
        }
        if (
            matching.get("enabled") is not True
            or matching.get("measured_counts_required") is not True
            or not baseline_ids.issubset(set(matching_baselines))
        ):
            self.fail("FIELD_INVALID", "$.audits.capacity_matching", "Capacity audit must cover both required baselines.")
        coverage = self.require_mapping(audits.get("claim_coverage"), "$.audits.claim_coverage")
        if coverage.get("enabled") is not True:
            self.fail("FIELD_INVALID", "$.audits.claim_coverage.enabled", "Claim-coverage audit must be enabled.")

    def validate_claim_coverage(self) -> None:
        start = self.error_count
        experiments = self.require_list(self.config.get("experiments"), "$.experiments", nonempty=True)
        claims = self.require_list(self.config.get("claims"), "$.claims", nonempty=True)
        task_system_id = _as_mapping(self.config.get("task")).get("system_id")
        usage_mode = _as_mapping(self.config.get("usage")).get("mode")
        usage = _as_mapping(self.config.get("usage"))
        expected_horizons = (
            _as_list(_as_mapping(usage.get("transition")).get("rollout_horizons"))
            if usage_mode == "repeated_transition"
            else []
        )
        baseline_kind_by_id = {
            baseline.get("id"): baseline.get("kind")
            for baseline in _as_list(self.config.get("baselines"))
            if isinstance(baseline, Mapping)
        }
        known_baseline_ids = set(baseline_kind_by_id)
        known_audit_ids = set(_as_mapping(self.config.get("audits")))

        experiment_by_id: dict[str, Mapping[str, Any]] = {}
        metrics_by_experiment: dict[str, dict[str, str]] = {}
        for index, raw in enumerate(experiments):
            path = f"$.experiments[{index}]"
            experiment = self.require_mapping(raw, path)
            allowed_experiment_keys = {
                "id",
                "system_id",
                "usage_mode",
                "split",
                "horizons",
                "metrics",
                "baselines",
                "audits",
                "tests_claims",
            }
            extra_experiment_keys = set(experiment) - allowed_experiment_keys
            missing_experiment_keys = allowed_experiment_keys - set(experiment)
            if extra_experiment_keys or missing_experiment_keys:
                self.fail(
                    "FIELD_INVALID",
                    path,
                    "Planned experiment keys mismatch; "
                    f"missing={sorted(missing_experiment_keys)}, "
                    f"extra={sorted(extra_experiment_keys)}.",
                )
            experiment_id = self.require_string(experiment.get("id"), f"{path}.id")
            if experiment_id in experiment_by_id:
                self.fail("FIELD_INVALID", f"{path}.id", "Experiment IDs must be unique.")
            experiment_by_id[experiment_id] = experiment
            if experiment.get("system_id") != task_system_id:
                self.fail("SYSTEM_IDENTITY_MISMATCH", f"{path}.system_id", "Experiment system_id must match task.system_id.")
            if experiment.get("usage_mode") != usage_mode:
                self.fail("USAGE_MODE_CONTRACT_INVALID", f"{path}.usage_mode", "Experiment usage_mode must match usage.mode.")
            split = self.require_string(experiment.get("split"), f"{path}.split")
            del split
            horizons = self.require_list(experiment.get("horizons"), f"{path}.horizons")
            if (
                any(not _is_positive_int(horizon) for horizon in horizons)
                or len(horizons) != len(set(horizons))
                or horizons != sorted(horizons)
                or horizons != expected_horizons
            ):
                self.fail(
                    "CLAIM_HORIZON_MISMATCH",
                    f"{path}.horizons",
                    "Experiment horizons must exactly match the selected use-mode contract.",
                )

            metric_entries = self.require_list(
                experiment.get("metrics"), f"{path}.metrics", nonempty=True
            )
            metric_map: dict[str, str] = {}
            for metric_index, raw_metric in enumerate(metric_entries):
                metric_path = f"{path}.metrics[{metric_index}]"
                metric = self.require_mapping(raw_metric, metric_path)
                if set(metric) != {"id", "direction"}:
                    self.fail(
                        "FIELD_INVALID",
                        metric_path,
                        "Metric entries may contain only id and direction.",
                    )
                metric_id = self.require_string(metric.get("id"), f"{metric_path}.id")
                direction = metric.get("direction")
                if direction not in METRIC_DIRECTIONS:
                    self.fail(
                        "FIELD_INVALID",
                        f"{metric_path}.direction",
                        f"Expected one of {sorted(METRIC_DIRECTIONS)}.",
                    )
                if metric_id in metric_map:
                    self.fail("FIELD_INVALID", f"{metric_path}.id", "Metric IDs must be unique within an experiment.")
                if metric_id and isinstance(direction, str):
                    metric_map[metric_id] = direction
            metrics_by_experiment[experiment_id] = metric_map

            experiment_baselines = self.require_string_list(
                experiment.get("baselines"), f"{path}.baselines", nonempty=True
            )
            unknown_baselines = set(experiment_baselines) - known_baseline_ids
            if unknown_baselines:
                self.fail(
                    "CLAIM_BASELINE_UNCOVERED",
                    f"{path}.baselines",
                    f"Unknown baseline IDs: {sorted(unknown_baselines)}.",
                )
            experiment_audits = self.require_string_list(
                experiment.get("audits"), f"{path}.audits", nonempty=True
            )
            unknown_audits = set(experiment_audits) - known_audit_ids
            if unknown_audits:
                self.fail(
                    "CLAIM_AUDIT_UNCOVERED",
                    f"{path}.audits",
                    f"Unknown audit IDs: {sorted(unknown_audits)}.",
                )
            self.require_string_list(
                experiment.get("tests_claims"), f"{path}.tests_claims", nonempty=True
            )

        claim_by_id: dict[str, Mapping[str, Any]] = {}
        claim_allowed_keys = {
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
        for index, raw in enumerate(claims):
            path = f"$.claims[{index}]"
            claim = self.require_mapping(raw, path)
            extra_keys = set(claim) - claim_allowed_keys
            missing_keys = claim_allowed_keys - set(claim)
            if extra_keys or missing_keys:
                self.fail(
                    "FIELD_INVALID",
                    path,
                    f"Structured claim keys mismatch; missing={sorted(missing_keys)}, extra={sorted(extra_keys)}.",
                )
            claim_id = self.require_string(claim.get("id"), f"{path}.id")
            if claim_id in claim_by_id:
                self.fail("FIELD_INVALID", f"{path}.id", "Claim IDs must be unique.")
            claim_by_id[claim_id] = claim
            if claim.get("status") != "planned":
                self.fail("FIELD_INVALID", f"{path}.status", "Design-time claims must have status 'planned'.")
            if claim.get("model_id") != _as_mapping(self.config.get("model")).get("id"):
                self.fail("FIELD_INVALID", f"{path}.model_id", "Claim model_id must name the reference model.")
            self.require_string(claim.get("metric_id"), f"{path}.metric_id")
            if claim.get("direction") not in METRIC_DIRECTIONS:
                self.fail("FIELD_INVALID", f"{path}.direction", f"Expected one of {sorted(METRIC_DIRECTIONS)}.")
            self.require_string(claim.get("split"), f"{path}.split")
            claim_horizons = self.require_list(claim.get("horizons"), f"{path}.horizons")
            if claim_horizons != expected_horizons:
                self.fail(
                    "CLAIM_HORIZON_MISMATCH",
                    f"{path}.horizons",
                    "Claim horizons must exactly match the selected use-mode contract.",
                )
            if claim.get("usage_mode") != usage_mode:
                self.fail(
                    "USAGE_MODE_CONTRACT_INVALID",
                    f"{path}.usage_mode",
                    "Claim usage_mode must match usage.mode.",
                )
            claim_baselines = self.require_string_list(
                claim.get("baseline_ids"), f"{path}.baseline_ids", nonempty=True
            )
            kinds_covered = {baseline_kind_by_id.get(item) for item in claim_baselines}
            if not REQUIRED_BASELINE_KINDS.issubset(kinds_covered):
                self.fail(
                    "CLAIM_BASELINE_UNCOVERED",
                    f"{path}.baseline_ids",
                    "Claim must require both standard_jepa and unconstrained_multihead baselines.",
                )
            unknown_claim_baselines = set(claim_baselines) - known_baseline_ids
            if unknown_claim_baselines:
                self.fail(
                    "CLAIM_BASELINE_UNCOVERED",
                    f"{path}.baseline_ids",
                    f"Unknown baseline IDs: {sorted(unknown_claim_baselines)}.",
                )
            claim_audits = self.require_string_list(
                claim.get("audit_ids"), f"{path}.audit_ids", nonempty=True
            )
            unknown_claim_audits = set(claim_audits) - known_audit_ids
            if unknown_claim_audits:
                self.fail(
                    "CLAIM_AUDIT_UNCOVERED",
                    f"{path}.audit_ids",
                    f"Unknown audit IDs: {sorted(unknown_claim_audits)}.",
                )
            if set(claim_audits) != REQUIRED_CLAIM_AUDITS:
                self.fail(
                    "CLAIM_AUDIT_UNCOVERED",
                    f"{path}.audit_ids",
                    "Every planned model comparison must require geometry, factor activity, "
                    "online-encoder activity, target-leakage, and capacity-matching audits.",
                )
            self.require_string_list(
                claim.get("experiment_ids"), f"{path}.experiment_ids", nonempty=True
            )

        for claim_index, raw_claim in enumerate(claims):
            if not isinstance(raw_claim, Mapping):
                continue
            claim_id = raw_claim.get("id")
            if not isinstance(claim_id, str) or not claim_id:
                continue
            required_experiments = _string_set(raw_claim.get("experiment_ids"))
            required_baselines = _string_set(raw_claim.get("baseline_ids"))
            required_audits = _string_set(raw_claim.get("audit_ids"))
            for experiment_id in sorted(required_experiments):
                experiment = experiment_by_id.get(experiment_id)
                if experiment is None:
                    self.fail(
                        "CLAIM_EXPERIMENT_MISSING",
                        f"$.claims[{claim_index}].experiment_ids",
                        f"Required experiment {experiment_id!r} does not exist.",
                    )
                    continue
                if claim_id not in _string_set(experiment.get("tests_claims")):
                    self.fail(
                        "CLAIM_RECIPROCITY_MISSING",
                        f"$.experiments[{experiment_id}].tests_claims",
                        f"Experiment does not reciprocally list claim {claim_id!r}.",
                    )
                metric_id = raw_claim.get("metric_id")
                metric_direction = metrics_by_experiment.get(experiment_id, {}).get(metric_id)
                if metric_direction is None:
                    self.fail(
                        "CLAIM_METRIC_UNCOVERED",
                        f"$.experiments[{experiment_id}].metrics",
                        f"Missing structured metric {metric_id!r}.",
                    )
                elif metric_direction != raw_claim.get("direction"):
                    self.fail(
                        "CLAIM_DIRECTION_MISMATCH",
                        f"$.experiments[{experiment_id}].metrics",
                        "Experiment metric direction does not match the claim.",
                    )
                if experiment.get("split") != raw_claim.get("split"):
                    self.fail(
                        "CLAIM_SPLIT_MISMATCH",
                        f"$.experiments[{experiment_id}].split",
                        "Experiment split does not match the claim.",
                    )
                if experiment.get("horizons") != raw_claim.get("horizons"):
                    self.fail(
                        "CLAIM_HORIZON_MISMATCH",
                        f"$.experiments[{experiment_id}].horizons",
                        "Experiment horizons do not match the claim.",
                    )
                if experiment.get("usage_mode") != raw_claim.get("usage_mode"):
                    self.fail(
                        "USAGE_MODE_CONTRACT_INVALID",
                        f"$.experiments[{experiment_id}].usage_mode",
                        "Experiment usage_mode does not match the claim.",
                    )
                experiment_baselines = _string_set(experiment.get("baselines"))
                if experiment_baselines != required_baselines:
                    self.fail(
                        "CLAIM_BASELINE_UNCOVERED",
                        f"$.experiments[{experiment_id}].baselines",
                        "Experiment baseline IDs must exactly match the structured claim.",
                    )
                experiment_audits = _string_set(experiment.get("audits"))
                if experiment_audits != required_audits:
                    self.fail(
                        "CLAIM_AUDIT_UNCOVERED",
                        f"$.experiments[{experiment_id}].audits",
                        "Experiment audit IDs must exactly match the structured claim.",
                    )

        for experiment_id, experiment in experiment_by_id.items():
            for claim_id in _string_set(experiment.get("tests_claims")):
                if claim_id not in claim_by_id:
                    self.fail(
                        "CLAIM_EXPERIMENT_MISSING",
                        f"$.experiments[{experiment_id}].tests_claims",
                        f"Experiment references unknown claim {claim_id!r}.",
                    )
                elif experiment_id not in _string_set(
                    claim_by_id[claim_id].get("experiment_ids")
                ):
                    self.fail(
                        "CLAIM_RECIPROCITY_MISSING",
                        f"$.experiments[{experiment_id}].tests_claims",
                        f"Claim {claim_id!r} does not reciprocally list this experiment.",
                    )

        self.group_pass(
            start,
            "CLAIM_COVERAGE_COMPLETE",
            "$.claims",
            "Every structured planned claim is reciprocally covered with matching metric direction, split, horizons, baselines, audits, and use mode.",
        )

    def validate_outputs(self) -> None:
        start = self.error_count
        outputs = self.require_mapping(self.config.get("outputs"), "$.outputs")
        format_list = self.require_string_list(
            outputs.get("config_formats"), "$.outputs.config_formats", nonempty=True
        )
        formats = set(format_list)
        if not formats or not formats.issubset({"json", "yaml"}):
            self.fail("FIELD_INVALID", "$.outputs.config_formats", "Use a non-empty subset of json and yaml.")
        scaffold = self.require_mapping(outputs.get("scaffold"), "$.outputs.scaffold")
        if scaffold.get("language") != "python":
            self.fail("FIELD_INVALID", "$.outputs.scaffold.language", "Only the Python scaffold is supported.")
        if scaffold.get("include_training_loop") is not False:
            self.fail(
                "OUTPUT_TRAINING_FORBIDDEN",
                "$.outputs.scaffold.include_training_loop",
                "The task designer cannot generate a training loop.",
            )
        self.group_pass(
            start,
            "OUTPUT_BOUNDARY_VALID",
            "$.outputs",
            "Outputs are limited to configuration and non-training code skeletons.",
        )

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": self.error_count == 0,
            "summary": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "passes": sum(check["status"] == "pass" for check in self.checks),
            },
            "checks": self.checks,
            "disclaimer": (
                "A valid design has complete declared controls and experiment coverage; "
                "it does not prove coordinate activity, semantics, model quality, or claim support."
            ),
        }


def validate_config(config: Any) -> dict[str, Any]:
    """Return a stable machine-readable validation report."""

    return DesignValidator(config).validate()


def load_config(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read {path}: {exc}") from exc
    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ConfigLoadError(
                    "YAML input requires PyYAML; use JSON for dependency-free validation."
                ) from exc
            return yaml.safe_load(text)
        return json.loads(text)
    except ConfigLoadError:
        raise
    except Exception as exc:
        raise ConfigLoadError(f"Cannot parse {path}: {exc}") from exc


def _serialize_report(report: Mapping[str, Any], pretty: bool) -> str:
    return json.dumps(
        report,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=False,
        ensure_ascii=False,
    ) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Design configuration (.json, or .yaml with PyYAML).")
    parser.add_argument("--output", type=Path, help="Also write the JSON validation report to this path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        report = validate_config(config)
        rendered = _serialize_report(report, args.pretty)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (ConfigLoadError, OSError) as exc:
        print(f"validate_design: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(rendered)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
