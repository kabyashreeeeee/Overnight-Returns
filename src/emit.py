"""Assemble the three submission CSVs from a scored frame, per Section 3 of the brief.

Statistics are computed from the *rounded* values that are actually written, so
rank ties in the CSV match the ties the grader sees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as MET

PRED_COLS = ["pred_date", "target_date", "symbol", "pred_magnitude_pct", "pred_direction",
             "conf_direction", "conf_magnitude", "split"]
ACT_COLS = ["pred_date", "target_date", "symbol", "actual_return_pct", "actual_direction",
            "actual_magnitude_pct", "universe_mean_pct"]


def write_outputs(frame: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = frame.sort_values(["pred_date", "symbol"]).reset_index(drop=True).copy()

    f["pred_magnitude_pct"] = np.round(np.maximum(f.pred_magnitude_pct, 0.0), 4)
    f["conf_direction"] = np.round(np.clip(f.conf_direction, 0.5, 1.0), 6)
    f["conf_magnitude"] = np.round(np.clip(f.conf_magnitude, 0.0, 1.0), 6)
    f["pred_direction"] = f.pred_direction.astype(int)
    f["actual_direction"] = np.where(f.actual_return_pct < 0, -1, 1).astype(int)
    f["actual_magnitude_pct"] = f.actual_return_pct.abs()
    f["universe_mean_pct"] = f.groupby("pred_date").actual_return_pct.transform("mean")

    for c in ["pred_date", "target_date"]:
        f[c] = pd.to_datetime(f[c]).dt.strftime("%Y-%m-%d")

    preds = f[PRED_COLS].copy()
    acts = f[ACT_COLS].copy()
    # the brief specifies four decimals for pred_magnitude_pct; to_csv would strip
    # trailing zeros, so format these columns as fixed-width strings explicitly
    preds["pred_magnitude_pct"] = preds.pred_magnitude_pct.map(lambda v: f"{v:.4f}")
    preds["conf_direction"] = preds.conf_direction.map(lambda v: f"{v:.6f}")
    preds["conf_magnitude"] = preds.conf_magnitude.map(lambda v: f"{v:.6f}")
    for c in ["actual_return_pct", "actual_magnitude_pct", "universe_mean_pct"]:
        acts[c] = acts[c].map(lambda v: f"{v:.10f}")
    preds.to_csv(out_dir / "predictions.csv", index=False)
    acts.to_csv(out_dir / "actuals.csv", index=False)

    # re-read the exact serialized interface before scoring
    p = pd.read_csv(out_dir / "predictions.csv", parse_dates=["pred_date", "target_date"])
    a = pd.read_csv(out_dir / "actuals.csv", parse_dates=["pred_date", "target_date"])
    scored = p.merge(a, on=["pred_date", "target_date", "symbol"], validate="1:1")
    scored = MET.add_vol_baseline(scored)
    stats = MET.statistics_table(scored)
    stats.to_csv(out_dir / "statistics.csv", index=False)

    return {"rows": len(preds), "statistics_rows": len(stats),
            "splits": sorted(preds.split.unique())}


def validate_schema(out_dir: Path) -> list[str]:
    """Mechanical checks mirroring how the grader reads the files."""
    problems = []
    p = pd.read_csv(out_dir / "predictions.csv")
    a = pd.read_csv(out_dir / "actuals.csv")
    s = pd.read_csv(out_dir / "statistics.csv")

    if list(p.columns) != PRED_COLS:
        problems.append(f"predictions columns {list(p.columns)}")
    if list(a.columns) != ACT_COLS:
        problems.append(f"actuals columns {list(a.columns)}")
    if list(s.columns) != ["split", "scope", "metric", "value", "n_obs"]:
        problems.append(f"statistics columns {list(s.columns)}")
    if p.isna().any().any() or a.isna().any().any():
        problems.append("nulls present")
    if not np.isfinite(p[["pred_magnitude_pct", "conf_direction", "conf_magnitude"]].to_numpy()).all():
        problems.append("non-finite predictions")
    if not set(p.pred_direction.unique()) <= {-1, 1}:
        problems.append("pred_direction not in {-1,+1}")
    if (p.pred_magnitude_pct < 0).any():
        problems.append("negative magnitude")
    if not p.conf_direction.between(0.5, 1.0).all():
        problems.append("conf_direction outside [0.5,1]")
    if not p.conf_magnitude.between(0.0, 1.0).all():
        problems.append("conf_magnitude outside [0,1]")
    if p.duplicated(["pred_date", "symbol"]).any():
        problems.append("duplicate keys")
    if len(p) != len(a) or not p[["pred_date", "symbol"]].equals(a[["pred_date", "symbol"]]):
        problems.append("prediction/actual keys differ")
    n_splits = p.split.nunique()
    if len(s) != 41 * n_splits:
        problems.append(f"statistics rows {len(s)} != 41 x {n_splits}")
    return problems
