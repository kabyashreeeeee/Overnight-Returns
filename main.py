"""Single entry point. One documented command reproduces all three CSVs.

    python3 main.py --config config.yaml --mode reproduce-all

Stages run in dependency order. The test block is dropped at load during every
research stage and is only reachable through `run_final.py`, which itself refuses
to start unless the frozen selection manifest exists.
"""
from __future__ import annotations

import argparse
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("build_panel.py", ["--mode", "research"], "canonical panel, integrity report, OOF folds"),
    ("run_magnitude.py", [], "Branch A: conditional-mean magnitude tournament"),
    ("run_direction.py", [], "Branch B: market + residual direction, quantile support model"),
    ("run_confidence.py", [], "Branches C and D: direction confidence, magnitude reliability"),
    ("run_freeze.py", [], "freeze every selection"),
    ("run_final.py", [], "single guarded test evaluation + CSV emission"),
    ("make_report.py", [], "research.pdf"),
    ("package.py", [], "verified submission folder and zip"),
]


def run_stage(script: str, extra: list[str], config: Path) -> None:
    argv = [script, "--config", str(config), *extra]
    if script in {"make_report.py", "package.py"}:
        argv = [script]
    sys.argv = argv
    runpy.run_path(str(ROOT / script), run_name="__main__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--mode", choices=["reproduce-all", "research-only"], default="reproduce-all")
    args = ap.parse_args()

    stages = STAGES if args.mode == "reproduce-all" else STAGES[:5]
    for i, (script, extra, what) in enumerate(stages, 1):
        print(f"\n{'='*72}\n[{i}/{len(stages)}] {script} — {what}\n{'='*72}", flush=True)
        t0 = time.time()
        run_stage(script, extra, args.config)
        print(f"[{i}/{len(stages)}] done in {time.time()-t0:.1f}s", flush=True)
    print("\nall stages complete")


if __name__ == "__main__":
    main()
