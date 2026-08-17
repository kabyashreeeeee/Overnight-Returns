"""Freeze manifests and checksums. The final-test path refuses to run without them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: object) -> str:
    """Hash a semantic object canonically, independent of key order/whitespace."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


EXCLUDED_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "outputs", "submission", "archive", ".git",
}


def sha256_tree(root: Path, patterns=("*.py", "*.yaml", "*.md", "*.txt")) -> dict:
    out = {}
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if EXCLUDED_PARTS.intersection(p.relative_to(root).parts):
                continue
            out[str(p.relative_to(root))] = sha256_file(p)
    return out


def sha256_files(root: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in names}


def verify_hashes(actual_root: Path, expected: dict[str, str], what: str) -> None:
    missing = sorted(name for name in expected if not (actual_root / name).is_file())
    changed = sorted(
        name for name, digest in expected.items()
        if (actual_root / name).is_file() and sha256_file(actual_root / name) != digest
    )
    if missing or changed:
        raise RuntimeError(
            f"refusing to continue: frozen {what} changed; missing={missing}, changed={changed}"
        )


def verify_code_tree(root: Path, expected: dict[str, str]) -> None:
    current = sha256_tree(root)
    missing = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    changed = sorted(name for name in set(expected) & set(current) if expected[name] != current[name])
    if missing or added or changed:
        raise RuntimeError(
            "refusing to continue: code differs from the frozen tree; "
            f"missing={missing}, added={added}, changed={changed}"
        )


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_frozen(
    path: Path,
    what: str,
    *,
    code_root: Path | None = None,
    artifact_root: Path | None = None,
) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"refusing to run the final test path: {what} is not frozen ({path} missing). "
            "Run the research stages first."
        )
    manifest = read_manifest(path)
    if code_root is not None:
        verify_code_tree(code_root, manifest["code_sha256"])
    if artifact_root is not None:
        verify_hashes(
            artifact_root,
            manifest.get("research_artifact_sha256", {}),
            "research artifacts",
        )
    return manifest
