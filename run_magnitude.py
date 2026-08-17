"""Stage 2: Branch A tournament. OOF inside train, one look at official validation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import magnitude as M
from src import features as F
from src import oof as O


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parent
    out = (root / cfg["paths"]["output_dir"]).resolve()
    seed = int(cfg["seed"])

    panel = pd.read_parquet(out / "panel.parquet")
    cols = pd.read_csv(out / "feature_columns.csv")["feature"].tolist()
    folds_df = pd.read_csv(out / "oof_folds.csv", parse_dates=["train_end", "predict_start", "predict_end"])
    folds = [O.Fold(int(r.fold), r.train_end, r.predict_start, r.predict_end, [])
             for r in folds_df.itertuples()]

    train = panel[panel.split == "train"].copy()
    valid = panel[panel.split == "valid"].copy()
    cats = sorted(panel.symbol.unique())
    cand = M.candidates(cfg["magnitude"])
    print(f"train {len(train)} | valid {len(valid)} | features {len(cols)} | candidates {len(cand)}")

    # ---------------- OOF inside the training block ----------------
    oof_store = {}
    for name, factory in cand.items():
        t0 = time.time()
        preds = np.full(len(train), np.nan)
        for f in folds:
            tr = (train.pred_date <= f.train_end).to_numpy()
            pr = train.pred_date.between(f.predict_start, f.predict_end).to_numpy()
            if tr.sum() < 5000 or pr.sum() == 0:
                continue
            fn = factory(train[tr], cols, cats, cfg["magnitude"], seed)
            fold_pred = np.asarray(fn(train[pr]), dtype=float)
            if not np.isfinite(fold_pred).all():
                raise AssertionError(f"{name} emitted non-finite OOF predictions in fold {f.index}")
            preds[pr] = fold_pred
        oof_store[name] = preds
        cov = np.isfinite(preds).mean()
        print(f"  OOF {name:24s} coverage {cov:5.1%}  {time.time()-t0:6.1f}s", flush=True)

    oof = pd.DataFrame(oof_store)
    oof.insert(0, "pred_date", train.pred_date.to_numpy())
    oof.insert(1, "symbol", train.symbol.to_numpy())
    oof["actual_magnitude_pct"] = train.actual_magnitude_pct.to_numpy()
    oof.to_parquet(out / "magnitude_oof.parquet", index=False)

    # ---------------- constrained non-negative blend on OOF ----------------
    mean_models = [c for c in cand if not c.startswith("S_") and c != "M0_trailing20"]
    sub = oof.dropna(subset=mean_models + ["actual_magnitude_pct"])
    A = sub[mean_models].to_numpy()
    y = sub.actual_magnitude_pct.to_numpy()
    from scipy.optimize import nnls
    w, _ = nnls(A, y)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(mean_models)) / len(mean_models)
    blend_w = dict(zip(mean_models, np.round(w, 4)))
    print(f"\nOOF non-negative blend weights: {blend_w}")

    # ---------------- refit on full train, score official validation once -------
    rows = []
    valid_preds = {}
    for name, factory in cand.items():
        fn = factory(train, cols, cats, cfg["magnitude"], seed)
        pv = np.asarray(fn(valid), dtype=float)
        if not np.isfinite(pv).all():
            raise AssertionError(f"{name} emitted non-finite validation predictions")
        valid_preds[name] = pv
    blend_valid = sum(blend_w[m] * valid_preds[m] for m in mean_models)
    valid_preds["M5_oof_blend"] = blend_valid
    blend_oof = sum(blend_w[m] * oof[m].to_numpy() for m in mean_models)
    oof_store["M5_oof_blend"] = blend_oof

    av = valid.actual_magnitude_pct.to_numpy(float)
    at = train.actual_magnitude_pct.to_numpy(float)
    for name, pv in valid_preds.items():
        po = oof_store[name]
        a0, b = M.affine_mean_calibration(po, at)          # calibration fit on OOF only
        pv_cal = M.apply_calibration(pv, a0, b)
        for tag, p in [("raw", pv), ("calibrated", pv_cal)]:
            gate = M.semantic_gate(p, av, cfg["magnitude"])
            err = np.abs(p - av)
            ic = np.mean([spearmanr(g.p, g.a).statistic for _, g in
                          pd.DataFrame({"d": valid.pred_date.to_numpy(), "p": p, "a": av})
                          .rename(columns={"p": "p", "a": "a"}).groupby("d")
                          if len(g) > 5 and g.p.nunique() > 1])
            vb = valid.actual_magnitude_pct.to_numpy()
            v20 = valid["mag_mean_20"].to_numpy(float)
            fin = np.isfinite(v20)
            r2 = 1 - np.sum((p[fin] - vb[fin]) ** 2) / np.sum((v20[fin] - vb[fin]) ** 2)
            rows.append({"model": name, "variant": tag,
                         "magnitude_score": 1 - err.sum() / av.sum(),
                         "mae": err.mean(), "rmse": np.sqrt(np.mean((p - av) ** 2)),
                         "rank_ic": ic, "r2_vs_vol": r2,
                         "calib_a0": a0, "calib_b": b, **gate})
    res = pd.DataFrame(rows).sort_values("magnitude_score", ascending=False)
    res.to_csv(out / "magnitude_validation.csv", index=False)
    pd.DataFrame([blend_w]).to_csv(out / "magnitude_blend_weights.csv", index=False)

    show = ["model", "variant", "magnitude_score", "mae", "rmse", "rank_ic", "r2_vs_vol",
            "mean_pred", "mean_actual", "mean_gap_pct", "calib_slope",
            "passes_mean_gate", "passes_slope_gate"]
    pd.set_option("display.width", 220)
    print("\n=== OFFICIAL VALIDATION (one look) ===")
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    gated = res[(res.passes_mean_gate) & (res.passes_slope_gate) & (~res.model.str.startswith("S_"))]
    print("\n=== SEMANTIC GATE SURVIVORS (may be called E[|r|]) ===")
    if len(gated):
        print(gated[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        best = gated.sort_values(["magnitude_score", "rmse"], ascending=[False, True]).iloc[0]
        print(f"\nSELECTED: {best.model} [{best.variant}]  magnitude_score={best.magnitude_score:.4f}"
              f"  mean_gap={best.mean_gap_pct:+.2f}%  slope={best.calib_slope:.3f}")
    else:
        print("  none passed; fall back to smallest |mean gap|")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    main(ap.parse_args().config)
