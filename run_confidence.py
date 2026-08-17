"""Stage 4: Branch C (direction confidence) and Branch D (magnitude reliability)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import confidence as C
from sklearn.isotonic import IsotonicRegression
from src import magnitude as M

def select_magnitude(out: Path) -> tuple[str, str]:
    """Declared rule: semantic mean gate, then highest magnitude_score, tie-break RMSE."""
    mag = pd.read_csv(out / "magnitude_validation.csv")
    gate = mag[(mag.passes_mean_gate) & (mag.passes_slope_gate)
               & (~mag.model.str.startswith("S_")) & (mag.model != "M0_trailing20")]
    if gate.empty:
        gate = mag[~mag.model.str.startswith("S_")].assign(
            g=lambda d: d.mean_gap_pct.abs()).sort_values("g")
    best = gate.sort_values(["magnitude_score", "rmse"], ascending=[False, True]).iloc[0]
    return best.model, best.variant


def ds(d, a):
    return float(np.sum(d * a) / np.sum(np.abs(a)))


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parent
    out = (root / cfg["paths"]["output_dir"]).resolve()
    seed = int(cfg["seed"])

    panel = pd.read_parquet(out / "panel.parquet")
    cols = pd.read_csv(out / "feature_columns.csv")["feature"].tolist()
    mag_oof = pd.read_parquet(out / "magnitude_oof.parquet")
    doof = pd.read_parquet(out / "direction_oof.parquet")
    dval = pd.read_parquet(out / "direction_valid.parquet")

    train = panel[panel.split == "train"].copy().reset_index(drop=True)
    valid = panel[panel.split == "valid"].copy().reset_index(drop=True)
    cats = sorted(panel.symbol.unique())

    # ---- magnitude: OOF preds + refit-on-train validation preds ----
    sel_name, sel_variant = select_magnitude(out)
    print(f"magnitude selected: {sel_name} [{sel_variant}]")
    bw = pd.read_csv(out / "magnitude_blend_weights.csv").iloc[0].to_dict()
    if sel_name == "M5_oof_blend":
        oof_raw = sum(w * mag_oof[m].to_numpy() for m, w in bw.items())
        fns = {m: M.candidates(cfg["magnitude"])[m](train, cols, cats, cfg["magnitude"], seed)
               for m in bw}
        val_raw = sum(w * fns[m](valid) for m, w in bw.items())
    else:
        oof_raw = mag_oof[sel_name].to_numpy()
        val_raw = M.candidates(cfg["magnitude"])[sel_name](
            train, cols, cats, cfg["magnitude"], seed)(valid)
    a0, b = M.affine_mean_calibration(oof_raw, mag_oof.actual_magnitude_pct.to_numpy())
    if sel_variant == "calibrated":
        mag_oof_pred = M.apply_calibration(oof_raw, a0, b)
        mag_val_pred = M.apply_calibration(val_raw, a0, b)
    else:
        mag_oof_pred, mag_val_pred = np.maximum(oof_raw, 0), np.maximum(val_raw, 0)
    print(f"  calib a0={a0:.4f} b={b:.4f} (applied: {sel_variant})")

    # ================= Branch C =================
    tr_c = C.build_direction_confidence_inputs(doof, panel, mag_oof_pred)
    va_c = C.build_direction_confidence_inputs(dval, panel, mag_val_pred)
    tr_c = tr_c.dropna(subset=["p_up_emitted", "mu_total"])
    print(f"\nBranch C: {len(tr_c)} OOF rows, {len(va_c)} validation rows")

    rows, fitted = [], {}
    for name, model in C.direction_confidence_candidates(seed).items():
        f = C.fit_direction_confidence(name, model, tr_c, seed)
        q = f(va_c)
        d, conf = C.emit(va_c.base_direction.to_numpy(), q)
        correct_final = (d == np.where(va_c.actual_return_pct >= 0, 1.0, -1.0)).astype(float)
        rep = C.calibration_report(conf, correct_final)
        av = va_c.actual_return_pct.to_numpy()
        base_ds = ds(d, av)
        w = 2 * conf - 1
        cds = float(np.sum(w * d * av) / np.sum(w * np.abs(av)))
        rows.append({"model": name, **rep, "lift": cds - base_ds,
                     "flips": int((q < 0.5).sum()), "pooled_ds": base_ds})
        fitted[name] = f
    res_c = pd.DataFrame(rows)
    res_c.to_csv(out / "direction_confidence_validation.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Branch C: direction confidence (validation, one look) ===")
    print(res_c.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    gate = res_c[(res_c.brier_skill >= 0) & (res_c.lift > 0) & (res_c.model != "C0_constant")]
    if len(gate):
        pick_c = gate.sort_values(["brier", "log_loss", "ece_10"]).iloc[0]["model"]
    else:
        pick_c = res_c.sort_values("brier").iloc[0]["model"]
    print(f"SELECTED (lowest Brier among lift>0 & brier_skill>=0): {pick_c}")

    q_val = fitted[pick_c](va_c)
    d_val, conf_val = C.emit(va_c.base_direction.to_numpy(), q_val)

    # ================= Branch D =================
    tr_d = doof.copy()
    tr_d["pred_magnitude_pct"] = mag_oof_pred
    tr_d["oof_abs_error"] = (tr_d.pred_magnitude_pct - tr_d.actual_magnitude_pct).abs()
    tr_d = tr_d.dropna(subset=["oof_abs_error", "pred_magnitude_pct"])
    tr_d = C.add_error_history(tr_d)
    tr_d = tr_d.merge(panel[["pred_date", "symbol"] + [c for c in C.CONF_MAG_FEATURES
                      if c in panel.columns and c not in tr_d.columns]],
                      on=["pred_date", "symbol"], how="left")

    va_d = dval.copy()
    va_d["pred_magnitude_pct"] = mag_val_pred
    va_d["oof_abs_error"] = (va_d.pred_magnitude_pct - va_d.actual_magnitude_pct).abs()
    hist = tr_d.groupby("symbol")[["err_hist_mean_20", "err_hist_std_20"]].last().reset_index()
    va_d = va_d.merge(hist, on="symbol", how="left")
    va_d = va_d.merge(panel[["pred_date", "symbol"] + [c for c in C.CONF_MAG_FEATURES
                      if c in panel.columns and c not in va_d.columns]],
                      on=["pred_date", "symbol"], how="left")

    err_v = va_d.oof_abs_error.to_numpy()
    mag_v = va_d.pred_magnitude_pct.to_numpy()
    rows_d = {}
    rows_d["D0_neg_magnitude"] = C.reliability_report(-mag_v, err_v, mag_v)

    scale_only = IsotonicRegression(out_of_bounds="clip")
    scale_only.fit(tr_d.pred_magnitude_pct, tr_d.oof_abs_error)
    pe = scale_only.predict(mag_v)
    tr_pe = scale_only.predict(tr_d.pred_magnitude_pct)
    rows_d["D1_scale_only"] = C.reliability_report(C.risk_to_confidence(pe, tr_pe), err_v, mag_v)

    conf_store = {"D0_neg_magnitude": C.risk_to_confidence(mag_v, -np.sort(-mag_v))}
    conf_store["D0_neg_magnitude"] = (-mag_v).argsort().argsort() / len(mag_v)
    for tag, tree in [("D2_two_stage_ridge", False), ("D3_two_stage_tree", True)]:
        pf, _ = C.fit_two_stage_reliability(tr_d, seed, use_tree=tree)
        pe = pf(va_d)
        tr_pe = pf(tr_d)
        conf = C.risk_to_confidence(pe, tr_pe)
        rows_d[tag] = C.reliability_report(conf, err_v, mag_v)
        conf_store[tag] = conf

    res_d = pd.DataFrame(rows_d).T.reset_index().rename(columns={"index": "model"})
    res_d.to_csv(out / "magnitude_confidence_validation.csv", index=False)
    print("\n=== Branch D: magnitude reliability (validation, one look) ===")
    print(res_d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Declared rule: highest conf_magnitude_score; within 0.01 tie-break on
    # within-magnitude-decile score, which is what separates reliability from scale.
    cand_d = res_d[res_d.model.isin(conf_store)].copy()
    top = cand_d.conf_magnitude_score.max()
    tie = cand_d[cand_d.conf_magnitude_score >= top - 0.01]
    pick_d = tie.sort_values("within_magnitude_decile", ascending=False).iloc[0]["model"]
    conf_mag_val = conf_store[pick_d]
    print(f"SELECTED: {pick_d} (top score {top:.4f}, tie band 0.01, "
          f"tie-broken on within-decile)")

    # ================= assemble validation predictions =================
    sub = pd.DataFrame({
        "pred_date": va_c.pred_date, "symbol": va_c.symbol,
        "pred_magnitude_pct": mag_val_pred,
        "pred_direction": d_val.astype(int),
        "conf_direction": conf_val,
        "conf_magnitude": conf_mag_val,
        "actual_return_pct": va_c.actual_return_pct,
        "actual_magnitude_pct": va_c.actual_magnitude_pct,
    })
    sub["universe_mean_pct"] = sub.groupby("pred_date").actual_return_pct.transform("mean")
    sub["split"] = "valid"
    numeric = ["pred_magnitude_pct", "pred_direction", "conf_direction",
               "conf_magnitude", "actual_return_pct", "actual_magnitude_pct",
               "universe_mean_pct"]
    if not np.isfinite(sub[numeric].to_numpy(float)).all():
        raise AssertionError("confidence stage emitted non-finite validation outputs")
    sub.to_parquet(out / "fresh_validation_predictions.parquet", index=False)
    print(f"\nwrote fresh_validation_predictions.parquet ({len(sub)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    main(ap.parse_args().config)
