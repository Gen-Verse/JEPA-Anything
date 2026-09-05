#!/usr/bin/env python3
"""Render an inspectable JEPA Anything code skeleton from a valid design."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from validate_design import ConfigLoadError, load_config, validate_config


TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / "assets" / "scaffold"


class ScaffoldError(RuntimeError):
    """Raised when the scaffold cannot be safely generated."""


def _python_repr(value: Any) -> str:
    """Stable repr for JSON-shaped values used in Python templates."""

    if isinstance(value, Mapping):
        items = ", ".join(
            f"{_python_repr(key)}: {_python_repr(item)}" for key, item in value.items()
        )
        return "{" + items + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_python_repr(item) for item in value) + "]"
    if isinstance(value, tuple):
        body = ", ".join(_python_repr(item) for item in value)
        if len(value) == 1:
            body += ","
        return "(" + body + ")"
    return repr(value)


def _template_values(config: Mapping[str, Any]) -> dict[str, str]:
    task = config["task"]
    observation = config["observation"]
    state = config["model"]["state"]
    usage = config["usage"]
    mode = usage["mode"]
    detail_key = {
        "terminal_readout": "readout",
        "repeated_transition": "transition",
        "factor_analysis": "analysis",
    }[mode]
    coordinates = tuple(factor["id"] for factor in state["factors"])
    baseline_ids = tuple(baseline["id"] for baseline in config["baselines"])
    baseline_kinds = tuple(baseline["kind"] for baseline in config["baselines"])
    project_name = f"jepa-task-{task['id']}".replace("_", "-")
    return {
        "PROJECT_NAME": project_name,
        "TASK_ID_REPR": _python_repr(task["id"]),
        "SYSTEM_ID_REPR": _python_repr(task["system_id"]),
        "MODEL_ID_REPR": _python_repr(config["model"]["id"]),
        "IDENTITY_KEYS_REPR": _python_repr(tuple(task["identity_keys"])),
        "STATE_D": str(state["d"]),
        "STATE_K": str(state["K"]),
        "STATE_R": str(state["r"]),
        "COORDINATES_REPR": _python_repr(coordinates),
        "USAGE_MODE_REPR": _python_repr(mode),
        "CONTEXT_INPUT_FIELDS_REPR": _python_repr(
            tuple(observation["adapter"]["context_input_fields"])
        ),
        "TARGET_INPUT_FIELDS_REPR": _python_repr(
            tuple(observation["adapter"]["target_input_fields"])
        ),
        "CONTEXT_SOURCES_REPR": _python_repr(tuple(observation["context"]["sources"])),
        "TARGET_FIELDS_REPR": _python_repr(tuple(observation["target"]["fields"])),
        "TARGET_DESCRIPTORS_REPR": _python_repr(
            tuple(observation["target_descriptor"]["fields"])
        ),
        "EXOGENOUS_INPUTS_REPR": _python_repr(
            tuple(observation["exogenous_inputs"]["fields"])
        ),
        "BASELINE_IDS_REPR": _python_repr(baseline_ids),
        "BASELINE_KINDS_REPR": _python_repr(baseline_kinds),
        "BASELINES_REPR": _python_repr(tuple(config["baselines"])),
        "CAPACITY_REPR": _python_repr(config["model"]["capacity"]),
        "STATE_ANALYSIS_REPR": _python_repr(state["analysis"]),
        "STATE_SYNTHESIS_REPR": _python_repr(state["synthesis"]),
        "LOSSES_REPR": _python_repr(config["losses"]),
        "AUDITS_REPR": _python_repr(config["audits"]),
        "CONTEXT_END": str(observation["context"]["window"]["end"]),
        "TARGET_START": str(observation["target"]["window"]["start"]),
        "TARGET_END": str(observation["target"]["window"]["end"]),
        "USAGE_DETAILS_REPR": _python_repr(usage[detail_key]),
        "EXPERIMENTS_REPR": _python_repr(config["experiments"]),
        "CLAIMS_REPR": _python_repr(config["claims"]),
    }


def _render_template(text: str, values: Mapping[str, str], source: Path) -> str:
    rendered = text
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z0-9_]+\}\}", rendered):
        raise ScaffoldError(f"Unresolved placeholder in template {source}.")
    return rendered


def _write_config(config: Mapping[str, Any], destination: Path, config_format: str) -> Path:
    config_dir = destination / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    if config_format == "json":
        path = config_dir / "design.json"
        rendered = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ScaffoldError(
                "YAML output requires PyYAML; use --config-format json for dependency-free generation."
            ) from exc
        path = config_dir / "design.yaml"
        rendered = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
    path.write_text(rendered, encoding="utf-8")
    return path.relative_to(destination)


def _ensure_destination_available(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise ScaffoldError(f"Destination exists and is not a directory: {destination}")
        try:
            next(destination.iterdir())
        except StopIteration:
            return
        raise ScaffoldError(f"Refusing to overwrite non-empty destination: {destination}")


def render_scaffold(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    destination: Path,
    *,
    config_format: str = "json",
) -> list[str]:
    """Render atomically into a missing or empty destination."""

    declared_formats = set(config["outputs"]["config_formats"])
    if config_format not in declared_formats:
        raise ScaffoldError(
            f"Requested config format {config_format!r} is not declared in outputs.config_formats."
        )
    _ensure_destination_available(destination)
    if not TEMPLATE_ROOT.is_dir():
        raise ScaffoldError(f"Template directory is missing: {TEMPLATE_ROOT}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=".jepa-scaffold-", dir=destination.parent))
    generated: list[str] = []
    try:
        values = _template_values(config)
        for template in sorted(TEMPLATE_ROOT.rglob("*.tmpl")):
            relative = template.relative_to(TEMPLATE_ROOT)
            output_relative = relative.with_suffix("")
            output = temp_path / output_relative
            output.parent.mkdir(parents=True, exist_ok=True)
            text = template.read_text(encoding="utf-8")
            output.write_text(_render_template(text, values, template), encoding="utf-8")
            generated.append(output_relative.as_posix())

        config_relative = _write_config(config, temp_path, config_format)
        generated.append(config_relative.as_posix())
        report_path = temp_path / "validation-report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        generated.append("validation-report.json")

        manifest_path = temp_path / "scaffold-manifest.json"
        manifest = {
            "schema_version": config["schema_version"],
            "task_id": config["task"]["id"],
            "system_id": config["task"]["system_id"],
            "usage_mode": config["usage"]["mode"],
            "contains_training_loop": False,
            "files": sorted(generated + ["scaffold-manifest.json"]),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        generated.append("scaffold-manifest.json")

        if destination.exists():
            destination.rmdir()
        temp_path.replace(destination)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return sorted(generated)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Validated design configuration.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Missing or empty destination directory.")
    parser.add_argument(
        "--config-format",
        choices=("json", "yaml"),
        default="json",
        help="Normalized configuration format in the generated skeleton.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw = load_config(args.config)
        report = validate_config(raw)
        if not report["valid"]:
            sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            return 1
        if not isinstance(raw, Mapping):
            raise ScaffoldError("Validated root is not an object.")
        files = render_scaffold(
            raw,
            report,
            args.output_dir,
            config_format=args.config_format,
        )
    except (ConfigLoadError, ScaffoldError, OSError) as exc:
        print(f"generate_scaffold: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "generated": True,
                "output_dir": str(args.output_dir),
                "files": files,
                "contains_training_loop": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
