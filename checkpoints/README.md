# Checkpoint manifests

This directory is metadata-only. Binary model weights are not committed here.
[`manifest.json`](manifest.json) records whether an artifact exists, which
recipe and design would produce it, the evidence boundary, and the hashes a
consumer must verify before loading any future artifact.

The current entry has `availability: "not_published"`, an empty `files` list,
and `training.executed: false`. It is a provenance record for a planned fixture,
not a downloadable checkpoint and not evidence of a result.

## Integrity policy

- Hash algorithm: SHA-256 over the exact file bytes.
- Encoding: 64 lowercase hexadecimal characters.
- Published artifacts must list every file, byte size, media type, and SHA-256.
  Each `files[].path` is resolved beneath a caller-supplied local artifact root;
  paths that escape that root are rejected.
- A `not_published` entry must have no artifact files and no fabricated digest.
- Source recipe, compiled design, and manifest schema hashes are pinned so a
  future producer can identify the exact interface contract.
- A changed source file requires updating its digest and provenance before a
  checkpoint can be published.

Validate the manifest with any JSON Schema 2020-12 implementation using
[`manifest.schema.json`](manifest.schema.json). The included verifier checks the
schema/source hashes, publication boundary, and absence of committed binary
weights without third-party dependencies. For every entry marked `published`,
it also opens each listed local file and verifies both byte size and exact-byte
SHA-256. It does not retrieve an `immutable_uri` or download any artifact:

```bash
python3 -m json.tool checkpoints/manifest.json >/dev/null
python3 -m json.tool checkpoints/manifest.schema.json >/dev/null
python3 checkpoints/verify_manifest.py --quiet
```

To verify a published bundle staged outside the repository, point the verifier
at the directory containing the listed relative paths:

```bash
python3 checkpoints/verify_manifest.py \
  --artifact-root /path/to/staged-checkpoint-bundle
```

The repository intentionally does not include `.pt`, `.pth`, `.ckpt`,
`.safetensors`, `.bin`, or equivalent weight files. If checkpoints are later
hosted externally, add an immutable URI and verified file entries to the
manifest, stage the bytes locally for verification, and pass their root with
`--artifact-root`; do not silently replace the planned entry.
