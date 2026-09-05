#!/usr/bin/env python3
"""Verify checkpoint manifest provenance and metadata-only repository policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


CHECKPOINT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CHECKPOINT_DIR.parent
DEFAULT_MANIFEST = CHECKPOINT_DIR / "manifest.json"
WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".joblib",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".weights",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_beneath(root: Path, relative_path: Any, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} path must be a non-empty string")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its root: {relative_path}") from exc
    return candidate


def resolve_repository_file(relative_path: Any) -> Path:
    return resolve_beneath(REPOSITORY_ROOT, relative_path, label="source")


def check_pinned_file(
    label: str,
    source: Any,
    checks: List[Dict[str, Any]],
) -> None:
    if not isinstance(source, dict):
        checks.append({"id": label, "passed": False, "detail": "source entry is not an object"})
        return
    expected = source.get("sha256")
    try:
        path = resolve_repository_file(source.get("path"))
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
    except (OSError, ValueError) as exc:
        checks.append({"id": label, "passed": False, "detail": str(exc)})
        return
    valid_digest = isinstance(expected, str) and bool(SHA256_PATTERN.fullmatch(expected))
    passed = exists and valid_digest and actual == expected
    checks.append(
        {
            "id": label,
            "passed": passed,
            "detail": (
                "Pinned exact-byte SHA-256 matches repository source."
                if passed
                else "Pinned repository source is missing or its SHA-256 does not match."
            ),
            "path": source.get("path"),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
    )


def check_local_artifact(
    label: str,
    artifact: Any,
    artifact_root: Path,
    checks: List[Dict[str, Any]],
) -> None:
    """Verify one published artifact from local bytes without fetching it."""

    if not isinstance(artifact, dict):
        checks.append(
            {"id": label, "passed": False, "detail": "artifact entry is not an object"}
        )
        return

    expected_sha256 = artifact.get("sha256")
    expected_bytes = artifact.get("bytes")
    try:
        path = resolve_beneath(artifact_root, artifact.get("path"), label="artifact")
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = sha256_file(path) if exists else None
    except (OSError, ValueError) as exc:
        checks.append({"id": label, "passed": False, "detail": str(exc)})
        return

    valid_sha256 = isinstance(expected_sha256, str) and bool(
        SHA256_PATTERN.fullmatch(expected_sha256)
    )
    valid_bytes = (
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and expected_bytes > 0
    )
    valid_media_type = (
        isinstance(artifact.get("media_type"), str)
        and bool(artifact["media_type"].strip())
    )
    passed = (
        exists
        and valid_sha256
        and valid_bytes
        and valid_media_type
        and actual_sha256 == expected_sha256
        and actual_bytes == expected_bytes
    )
    checks.append(
        {
            "id": label,
            "passed": passed,
            "detail": (
                "Local published artifact size and exact-byte SHA-256 match the manifest."
                if passed
                else "Local published artifact is missing or its size/SHA-256 does not match."
            ),
            "path": artifact.get("path"),
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "media_type": artifact.get("media_type"),
        }
    )


def verify(
    manifest: Mapping[str, Any],
    *,
    artifact_root: Path = REPOSITORY_ROOT,
) -> Dict[str, Any]:
    artifact_root = artifact_root.resolve()
    checks: List[Dict[str, Any]] = []
    checks.append(
        {
            "id": "manifest_contract",
            "passed": (
                manifest.get("schema_version")
                == "jepa-anything.checkpoint-manifest/v1"
                and manifest.get("publication_policy") == "metadata_only_in_git"
                and manifest.get("hash_policy")
                == {
                    "algorithm": "sha256",
                    "encoding": "lowercase_hex",
                    "scope": "exact_file_bytes",
                }
            ),
            "detail": "Manifest declares the supported schema, publication, and hash policy.",
        }
    )

    provenance = manifest.get("provenance", {})
    check_pinned_file("manifest_schema_sha256", provenance.get("manifest_schema"), checks)

    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list):
        checkpoints = []
        checks.append(
            {
                "id": "checkpoint_list",
                "passed": False,
                "detail": "checkpoints must be a list",
            }
        )

    for index, checkpoint in enumerate(checkpoints):
        prefix = f"checkpoint_{index}"
        if not isinstance(checkpoint, dict):
            checks.append(
                {"id": prefix, "passed": False, "detail": "checkpoint entry is not an object"}
            )
            continue
        check_pinned_file(f"{prefix}_recipe_sha256", checkpoint.get("sources", {}).get("recipe"), checks)
        check_pinned_file(
            f"{prefix}_compiled_design_sha256",
            checkpoint.get("sources", {}).get("compiled_design"),
            checks,
        )
        availability = checkpoint.get("availability")
        training = checkpoint.get("training", {})
        claims = checkpoint.get("claims", {})
        files = checkpoint.get("files")
        if availability == "not_published":
            boundary_valid = (
                checkpoint.get("immutable_uri") is None
                and files == []
                and training.get("executed") is False
                and training.get("steps") == 0
                and training.get("framework") is None
                and claims.get("scientific_results_claimed") is False
                and claims.get("result_claims") == []
            )
        else:
            boundary_valid = (
                availability == "published"
                and isinstance(checkpoint.get("immutable_uri"), str)
                and bool(checkpoint.get("immutable_uri"))
                and isinstance(files, list)
                and bool(files)
                and training.get("executed") is True
            )
            if isinstance(files, list):
                for file_index, artifact in enumerate(files):
                    check_local_artifact(
                        f"{prefix}_artifact_{file_index}_integrity",
                        artifact,
                        artifact_root,
                        checks,
                    )
        checks.append(
            {
                "id": f"{prefix}_availability_boundary",
                "passed": boundary_valid,
                "detail": "Availability is consistent with files, training provenance, and result-claim fields.",
                "availability": availability,
            }
        )

    binary_files = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in CHECKPOINT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )
    checks.append(
        {
            "id": "no_committed_weight_binaries",
            "passed": not binary_files,
            "detail": "Checkpoint source tree contains no recognized model-weight binaries.",
            "files": binary_files,
        }
    )

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": "jepa-anything.checkpoint-verification-report/v1",
        "status": "pass" if passed else "fail",
        "checks_passed": sum(1 for check in checks if check["passed"]),
        "checks_total": len(checks),
        "checks": checks,
        "artifact_root": artifact_root.as_posix(),
        "evidence_scope": "Integrity and publication-boundary verification only; no model result is evaluated.",
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help=(
            "Root containing files listed by published entries. The verifier "
            "never downloads artifacts; paths must resolve beneath this directory."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        report = verify(
            load_json(args.manifest.resolve()),
            artifact_root=args.artifact_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"manifest verification could not run: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
