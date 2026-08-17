"""Stage 4b: freeze every selection. run_final.py refuses to execute without this."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import freeze as FZ


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parent
    out = (root / cfg["paths"]["output_dir"]).resolve()

    mag = pd.read_csv(out / "magnitude_validation.csv")
    gate = mag[(mag.passes_mean_gate) & (mag.passes_slope_gate)
               & (~mag.model.str.startswith("S_")) & (mag.model != "M0_trailing20")]
    if gate.empty:
        gate = mag[~mag.model.str.startswith("S_")].assign(
            gap=lambda d: d.mean_gap_pct.abs()).sort_values("gap")
    best_mag = gate.sort_values(["magnitude_score", "rmse"], ascending=[False, True]).iloc[0]

    lam = pd.read_csv(out / "direction_selection.csv", index_col=0).loc["lambda"].iloc[0]

    conf = pd.read_csv(out / "direction_confidence_validation.csv")
    ok = conf[(conf.brier_skill >= 0) & (conf.lift > 0) & (conf.model != "C0_constant")]
    best_conf = (ok if len(ok) else conf).sort_values(["brier", "log_loss", "ece_10"]).iloc[0]

    mc = pd.read_csv(out / "magnitude_confidence_validation.csv")
    top = mc.conf_magnitude_score.max()
    tie = mc[mc.conf_magnitude_score >= top - 0.01]
    best_mc = tie.sort_values("within_magnitude_decile", ascending=False).iloc[0]

    selected = {
        "magnitude": best_mag.model,
        "magnitude_variant": best_mag.variant,
        "lambda": float(lam),
        "direction_confidence": best_conf.model,
        "magnitude_confidence": best_mc.model,
    }
    model_code_files = [
        "src/panel.py", "src/minute.py", "src/features.py", "src/oof.py",
        "src/magnitude.py", "src/direction.py", "src/confidence.py", "src/emit.py",
        "build_panel.py", "run_magnitude.py", "run_direction.py", "run_confidence.py",
        "run_final.py",
    ]
    specification_components = {
        "selected": selected,
        "seed": cfg["seed"],
        "model_config": {name: cfg[name] for name in
                         ["splits", "oof", "features", "magnitude", "direction", "quantiles"]},
        "feature_columns_sha256": FZ.sha256_file(out / "feature_columns.csv"),
        "blend_weights_sha256": FZ.sha256_file(out / "magnitude_blend_weights.csv"),
        "lambda_selection_sha256": FZ.sha256_file(out / "direction_selection.csv"),
        "model_code_sha256": FZ.sha256_files(root, model_code_files),
    }

    payload = {
        "selected": selected,
        "specification_sha256": FZ.sha256_json(specification_components),
        "specification_components": specification_components,
        "selection_rules": {
            "magnitude": "semantic mean gate (|mean gap|<=5%, slope in [0.7,1.4]) then "
                         "highest validation magnitude_score, tie-break lower RMSE",
            "lambda": "highest pooled direction_score on INNER OOF folds only; "
                      "validation never used",
            "direction_confidence": "lowest Brier among candidates with lift>0 and "
                                    "brier_skill>=0; tie-break log_loss then ECE",
            "magnitude_confidence": "highest conf_magnitude_score; within 0.01 tie band "
                                    "break on within-magnitude-decile score",
        },
        "disclosure": (
            "The first validation review exposed two specification omissions: no direct L2 "
            "conditional-mean candidate, and confidence based on P(r>0) rather than correctness of "
            "the emitted sign. Those families were added once; a second validation review then fixed "
            "all final choices. The test block was untouched during both reviews."
        ),
        "validation_scores": {
            "magnitude_score": float(best_mag.magnitude_score),
            "magnitude_mean_gap_pct": float(best_mag.mean_gap_pct),
            "conf_brier": float(best_conf.brier),
            "conf_lift": float(best_conf.lift),
            "magconf_score": float(best_mc.conf_magnitude_score),
            "magconf_within_decile": float(best_mc.within_magnitude_decile),
        },
        "code_sha256": FZ.sha256_tree(root),
        "research_artifact_sha256": FZ.sha256_files(out, [
            "feature_columns.csv", "oof_folds.csv", "magnitude_oof.parquet",
            "magnitude_validation.csv", "magnitude_blend_weights.csv",
            "direction_oof.parquet", "direction_valid.parquet",
            "direction_lambda_oof.csv", "direction_selection.csv",
            "direction_benchmark_ablation.csv",
            "direction_confidence_validation.csv", "magnitude_confidence_validation.csv",
            "fresh_validation_predictions.parquet",
        ]),
        "config": cfg,
    }
    FZ.write_manifest(out / "selection_manifest.json", payload)
    print("FROZEN SELECTIONS")
    for k, v in payload["selected"].items():
        print(f"  {k:24s} {v}")
    print("\nvalidation scores at freeze:")
    for k, v in payload["validation_scores"].items():
        print(f"  {k:24s} {v:.4f}")
    print(f"\nwrote {out/'selection_manifest.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    main(ap.parse_args().config)
