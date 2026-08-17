"""Assemble the submission folder and zip, then verify it independently."""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import freeze as FZ
from src.paths import PACKAGE_NAME, submission_dir
from src import verify as VERIFY

ROOT = Path(__file__).resolve().parent
SUB = submission_dir(ROOT)
CODE = ROOT if SUB == ROOT.parent else SUB / "code"

PIPELINE = ["main.py", "config.yaml", "requirements.txt", "README.md", "reproduction_audit.json",
            "build_panel.py", "run_magnitude.py", "run_direction.py", "run_confidence.py",
            "run_freeze.py", "run_final.py", "make_report.py", "compare_validation.py",
            "package.py"]
SRC = ["__init__.py", "panel.py", "features.py", "oof.py", "magnitude.py", "direction.py",
       "confidence.py", "metrics.py", "emit.py", "freeze.py", "minute.py", "paths.py",
       "verify.py"]
TESTS = ["test_pipeline.py"]


def prepare_submission_root(sub: Path, required: set[str], *, clean_extras: bool) -> None:
    """Clean the generated dev folder, but never delete inputs beside extracted code."""
    present = {p.name for p in sub.iterdir()}
    missing = required - present
    if missing:
        raise AssertionError(f"submission is missing required entries: {sorted(missing)}")
    if clean_extras:
        for name in present - required:
            p = sub / name
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
        present = {p.name for p in sub.iterdir()}
        if present != required:
            raise AssertionError(f"submission entries are {sorted(present)}")


def required_paths(sub: Path, required: set[str]):
    """Yield only deliverables; extracted raw data/cache are intentionally excluded."""
    transient_code_parts = {"outputs", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for name in sorted(required):
        path = sub / name
        if path.is_dir():
            yield path
            for child in sorted(path.rglob("*")):
                rel_parts = child.relative_to(path).parts
                if name == "code" and transient_code_parts.intersection(rel_parts):
                    continue
                yield child
        else:
            yield path


def main() -> None:
    selection = FZ.require_frozen(
        ROOT / "outputs" / "selection_manifest.json", "model selection",
        code_root=ROOT, artifact_root=ROOT / "outputs",
    )
    final = FZ.read_manifest(ROOT / "outputs" / "final_manifest.json")
    if final["selection_manifest_sha256"] != FZ.sha256_file(
        ROOT / "outputs" / "selection_manifest.json"
    ):
        raise RuntimeError("final manifest does not reference the current selection manifest")
    if final["specification_sha256"] != selection["specification_sha256"]:
        raise RuntimeError("final manifest does not reference the frozen specification")
    FZ.verify_code_tree(ROOT, final["code_sha256"])
    FZ.verify_hashes(SUB, final["artifact_sha256"], "final CSV outputs")
    audit_path = ROOT / "reproduction_audit.json"
    if final["reproduction_audit_sha256"] != FZ.sha256_file(audit_path):
        raise RuntimeError("final manifest does not reference the reproduction audit")
    audit = FZ.read_manifest(audit_path)
    FZ.verify_hashes(SUB, audit["artifact_sha256"], "clean-room reproduced CSVs")

    if CODE != ROOT:
        if CODE.exists():
            shutil.rmtree(CODE)
        (CODE / "src").mkdir(parents=True)
        (CODE / "tests").mkdir(parents=True)

        for f in PIPELINE:
            shutil.copy2(ROOT / f, CODE / f)
        for f in SRC:
            shutil.copy2(ROOT / "src" / f, CODE / "src" / f)
        for f in TESTS:
            shutil.copy2(ROOT / "tests" / f, CODE / "tests" / f)

    # Both manifests travel with the code as independently verifiable evidence.
    shutil.copy2(ROOT / "outputs" / "selection_manifest.json", CODE / "selection_manifest.json")
    shutil.copy2(ROOT / "outputs" / "final_manifest.json", CODE / "final_manifest.json")

    required = {"research.pdf", "predictions.csv", "actuals.csv", "statistics.csv", "code"}
    prepare_submission_root(SUB, required, clean_extras=(CODE != ROOT))

    zip_path = SUB.parent / f"{PACKAGE_NAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    skip = ("__pycache__", ".pyc", ".DS_Store", ".pytest_cache")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in required_paths(SUB, required):
            if any(s in str(p) for s in skip):
                continue
            z.write(p, p.relative_to(SUB.parent))
    interface_verification = VERIFY.verify_submission(SUB)
    zip_verification = VERIFY.verify_zip(zip_path, PACKAGE_NAME)

    manifest = {
        "zip_name": zip_path.name,
        "zip_sha256": FZ.sha256_file(zip_path),
        "artifact_sha256": {n: FZ.sha256_file(SUB / n)
                            for n in ["research.pdf", "predictions.csv", "actuals.csv",
                                      "statistics.csv"]},
        "code_sha256": FZ.sha256_tree(CODE),
        "selection_manifest_sha256": FZ.sha256_file(CODE / "selection_manifest.json"),
        "final_manifest_sha256": FZ.sha256_file(CODE / "final_manifest.json"),
        "selected": selection["selected"],
        "specification_sha256": selection["specification_sha256"],
        "reproduction_audit_sha256": FZ.sha256_file(CODE / "reproduction_audit.json"),
        "interface_verification": interface_verification,
        "zip_verification": zip_verification,
    }
    FZ.write_manifest(SUB.parent / f"{PACKAGE_NAME}_manifest.json", manifest)

    # ---- verify ----
    p = pd.read_csv(SUB / "predictions.csv")
    s = pd.read_csv(SUB / "statistics.csv")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    roots = sorted({n.split("/")[1] for n in names if len(n.split("/")) > 1 and n.split("/")[1]})
    cruft = [n for n in names if any(x in n for x in skip)]
    print(f"zip           : {zip_path}")
    print(f"sha256        : {manifest['zip_sha256']}")
    print(f"root entries  : {roots}")
    print(f"cruft in zip  : {cruft or 'none'}")
    print(f"predictions   : {len(p):,} rows, splits {sorted(p.split.unique())}")
    print(f"statistics    : {len(s)} rows")
    print(f"py files      : {sum(1 for n in names if n.endswith('.py'))}")
    print(f"size          : {zip_path.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
