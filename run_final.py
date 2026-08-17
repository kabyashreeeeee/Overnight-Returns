"""Stage 5: guarded final scoring after model selection is checksum-frozen.

Guarded by checksum-verified selection manifests. Train statistics are explicitly
in-sample; validation rows use train-only fits; test rows use a train+validation
refit, which is the information legitimately available at that point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import panel as P
from src import features as F
from src import magnitude as M
from src import direction as D
from src import confidence as C
from src import emit as E
from src import freeze as FZ
from src import minute as MIN
from src import metrics as MET
from src.paths import submission_dir
from src import verify as VERIFY


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parent
    out = (root / cfg["paths"]["output_dir"]).resolve()
    seed = int(cfg["seed"])

    sel = FZ.require_frozen(
        out / "selection_manifest.json", "model selection",
        code_root=root, artifact_root=out,
    )
    if FZ.sha256_json(sel["specification_components"]) != sel["specification_sha256"]:
        raise RuntimeError("frozen specification digest is inconsistent")
    print("frozen selections:", json.dumps(sel["selected"], indent=2))

    # ---------------- rebuild the panel including test ----------------
    daily = P.load_daily((root / cfg["paths"]["daily_dir"]).resolve(),
                         cfg["data"]["start_date"], cfg["data"]["end_date"])
    cal = P.master_calendar(daily)
    tgt = P.build_targets(daily, cal)
    bounds = P.assign_splits(cal, cfg["splits"]["train_end"], cfg["splits"]["valid_end"],
                             cfg["splits"]["embargo_sessions"])
    tgt = P.label_splits(tgt, bounds)
    minute = MIN.load_or_build(
        (root / cfg["paths"]["minute_dir"]).resolve(),
        (root / cfg["paths"]["minute_cache"]).resolve(),
        sorted(daily.symbol.unique()),
        int(cfg["features"].get("expected_minute_bars", 375)),
        bool(cfg["features"].get("rebuild_minute_cache", False)),
    )
    minute["pred_date"] = pd.to_datetime(minute["pred_date"]).dt.normalize()
    P.integrity_report(daily, tgt, minute).to_csv(out / "integrity_report.csv", index=False)
    feat = F.build_features(tgt, minute)
    feat = P.enforce_quarantine(feat, "final-test")

    minh = cfg["features"]["min_history_sessions"]
    panel = feat[feat.actual_return_pct.notna() & feat.split.notna()
                 & (feat.history_count >= minh)].copy().reset_index(drop=True)
    reconstructed = (panel.target_open / panel.close - 1.0) * 100.0
    if not np.allclose(panel.actual_return_pct, reconstructed, atol=1e-12):
        raise AssertionError("target reconstruction failed before final scoring")
    cols = pd.read_csv(out / "feature_columns.csv")["feature"].tolist()
    cats = sorted(panel.symbol.unique())
    panel["resid_target"] = panel.actual_return_pct - panel.groupby("pred_date").actual_return_pct.transform("mean")
    print("\nrows per split:", panel.groupby("split").size().to_dict())

    dev = panel[panel.split.isin(["train", "valid"])].copy()      # train+valid
    tr = panel[panel.split == "train"].copy()
    is_test = (panel.split == "test").to_numpy()
    is_dev = ~is_test

    mag_name = sel["selected"]["magnitude"]
    lam = float(sel["selected"]["lambda"])
    conf_name = sel["selected"]["direction_confidence"]
    magconf_name = sel["selected"]["magnitude_confidence"]

    # ---------------- magnitude ----------------
    mag_oof = pd.read_parquet(out / "magnitude_oof.parquet")
    mag_variant = sel["selected"]["magnitude_variant"]
    all_cand = M.candidates(cfg["magnitude"])

    def magnitude_fns(fit_frame):
        """Return a callable producing the selected magnitude forecast (raw scale)."""
        if mag_name == "M5_oof_blend":
            bw = pd.read_csv(out / "magnitude_blend_weights.csv").iloc[0].to_dict()
            fns = {m: all_cand[m](fit_frame, cols, cats, cfg["magnitude"], seed) for m in bw}
            return lambda fr: sum(w * fns[m](fr) for m, w in bw.items())
        f = all_cand[mag_name](fit_frame, cols, cats, cfg["magnitude"], seed)
        return lambda fr: f(fr)

    if mag_name == "M5_oof_blend":
        bw = pd.read_csv(out / "magnitude_blend_weights.csv").iloc[0].to_dict()
        oof_raw = sum(w * mag_oof[m].to_numpy() for m, w in bw.items())
    else:
        oof_raw = mag_oof[mag_name].to_numpy()
    a0, b = M.affine_mean_calibration(oof_raw, mag_oof.actual_magnitude_pct.to_numpy())
    cal = (lambda v: M.apply_calibration(v, a0, b)) if mag_variant == "calibrated" \
        else (lambda v: np.maximum(v, 0.0))

    fn_dev = magnitude_fns(tr)                                    # train-only -> dev rows
    fn_test = magnitude_fns(dev)                                  # train+valid -> test rows
    mag = np.empty(len(panel))
    mag[is_dev] = cal(fn_dev(panel[is_dev]))
    mag[is_test] = cal(fn_test(panel[is_test]))
    panel["pred_magnitude_pct"] = np.maximum(mag, 0.0)
    mag_oof_pred = cal(oof_raw)
    print(f"magnitude {mag_name} [{mag_variant}]: calib a0={a0:.4f} b={b:.4f}")

    # ---------------- direction ----------------
    def direction_for(fit_frame, target_frame):
        mf = D.market_frame(fit_frame)
        mm, alpha = D.fit_market(mf, cfg["direction"]["market_ridge_alphas"], seed)
        tf = D.market_frame(target_frame)
        mp = pd.Series(mm.predict(tf[D.MARKET_FEATURES]), index=tf.pred_date)
        mkt = target_frame.pred_date.map(mp).to_numpy()
        res = D.fit_residual(fit_frame, cols, cats, cfg["direction"], seed)(target_frame)
        qf = D.fit_quantiles(fit_frame, cols, cats, cfg["quantiles"], seed)
        q, cross = qf(target_frame)
        return mkt, res, q, cross

    mkt = np.empty(len(panel)); res = np.empty(len(panel))
    levels = cfg["quantiles"]["levels"]
    qmat = np.empty((len(panel), len(levels)))
    m_d, r_d, q_d, c_d = direction_for(tr, panel[is_dev])
    mkt[is_dev], res[is_dev], qmat[is_dev] = m_d, r_d, q_d
    m_t, r_t, q_t, c_t = direction_for(dev, panel[is_test])
    mkt[is_test], res[is_test], qmat[is_test] = m_t, r_t, q_t
    print(f"quantile crossings pre-rearrangement: dev {c_d:.2%}, test {c_t:.2%}")

    panel["mu_market"] = mkt
    panel["mu_residual"] = res
    panel["mu_total"] = mkt + lam * res
    panel["p_up"] = D.prob_up_from_quantiles(qmat, levels)
    panel = pd.concat([panel.reset_index(drop=True),
                       D.distribution_features(qmat, levels)], axis=1)

    # ---------------- direction confidence ----------------
    doof = pd.read_parquet(out / "direction_oof.parquet")
    tr_c = C.build_direction_confidence_inputs(doof, panel, mag_oof_pred).dropna(
        subset=["p_up_emitted", "mu_total", "abs_mu"])
    all_c = C.build_direction_confidence_inputs(
        panel[["pred_date", "symbol", "actual_return_pct", "resid_target",
               "actual_magnitude_pct", "mu_market", "mu_residual", "mu_total", "p_up",
               "q_iqr80", "q_iqr90", "q_skew", "q_median_abs", "q_left_tail", "q_right_tail"]],
        panel, panel.pred_magnitude_pct.to_numpy())
    cand_c = C.direction_confidence_candidates(seed)
    q_fn = C.fit_direction_confidence(conf_name, cand_c[conf_name], tr_c, seed)
    q_all = q_fn(all_c)
    d_final, conf_dir = C.emit(all_c.base_direction.to_numpy(), q_all)
    panel["pred_direction"] = d_final.astype(int)
    panel["conf_direction"] = conf_dir
    print(f"direction confidence {conf_name}: {(q_all < 0.5).sum()} flips")

    # ---------------- magnitude confidence ----------------
    tr_d = doof.copy()
    tr_d["pred_magnitude_pct"] = mag_oof_pred
    tr_d["oof_abs_error"] = (tr_d.pred_magnitude_pct - tr_d.actual_magnitude_pct).abs()
    tr_d = C.add_error_history(tr_d.dropna(subset=["oof_abs_error"]))
    extra = [c for c in C.CONF_MAG_FEATURES if c in panel.columns and c not in tr_d.columns]
    tr_d = tr_d.merge(panel[["pred_date", "symbol"] + extra], on=["pred_date", "symbol"], how="left")

    # Keep training diagnostics timestamp-valid: an early training row must not
    # receive the symbol's final OOF error history. Validation/test may use the
    # last training history because it is already known at those dates.
    all_d = C.attach_error_history(panel, tr_d)

    if magconf_name == "D0_neg_magnitude":
        conf_mag = (-all_d.pred_magnitude_pct.to_numpy()).argsort().argsort() / len(all_d)
    else:
        use_tree = magconf_name.endswith("tree")
        pf, _ = C.fit_two_stage_reliability(tr_d, seed, use_tree=use_tree)
        conf_mag = C.risk_to_confidence(pf(all_d), pf(tr_d))
    panel["conf_magnitude"] = conf_mag
    print(f"magnitude confidence {magconf_name}")

    robustness = MET.economic_robustness_table(panel)
    robustness.to_csv(out / "economic_robustness.csv", index=False)

    # ---------------- emit ----------------
    sub_dir = submission_dir(root)
    info = E.write_outputs(panel, sub_dir)
    problems = E.validate_schema(sub_dir)
    verification = VERIFY.verify_submission(sub_dir)
    audit_path = root / "reproduction_audit.json"
    reproduction_audit = FZ.read_manifest(audit_path)
    for name, expected in reproduction_audit["artifact_sha256"].items():
        if FZ.sha256_file(sub_dir / name) != expected:
            raise RuntimeError(f"{name} differs from the independently reproduced artifact")
    print(f"\nwrote CSVs to {sub_dir}: {info}")
    print("schema problems:", problems if problems else "NONE")

    stats = pd.read_csv(sub_dir / "statistics.csv")
    key = ["direction_score", "magnitude_score", "hit_rate", "r2_vs_vol", "rank_ic",
           "conf_direction_lift", "brier_skill", "ece_10", "conf_magnitude_score",
           "frac_stocks_beat_naive"]
    t = stats[stats.metric.isin(key)].pivot_table(index="metric", columns=["scope", "split"],
                                                  values="value")
    pd.set_option("display.width", 200)
    print("\n=== FRESH SYSTEM, ALL SPLITS ===")
    print(t.reindex(key).round(4).to_string())

    FZ.write_manifest(out / "final_manifest.json", {
        "selected": sel["selected"],
        "specification_sha256": sel["specification_sha256"],
        "selection_manifest_sha256": FZ.sha256_file(out / "selection_manifest.json"),
        "code_sha256": FZ.sha256_tree(root),
        "rows": info, "schema_problems": problems,
        "verification": verification,
        "artifact_sha256": {n: FZ.sha256_file(sub_dir / n)
                            for n in ["predictions.csv", "actuals.csv", "statistics.csv"]},
        "diagnostic_sha256": {
            "economic_robustness.csv": FZ.sha256_file(out / "economic_robustness.csv"),
            "integrity_report.csv": FZ.sha256_file(out / "integrity_report.csv"),
        },
        "quantile_crossings": {"dev": c_d, "test": c_t},
        "magnitude_calibration": {"a0": a0, "b": b},
        "reproduction_audit_sha256": FZ.sha256_file(audit_path),
        "test_governance": (
            "The fresh specification was frozen before its initial test evaluation. Subsequent "
            "executions reran that identical specification solely for checksum, rounding-interface, "
            "packaging, and clean-room verification; no feature, candidate, hyperparameter, ensemble "
            "weight, calibration rule, or emitted prediction changed. The later choice between this "
            "and the earlier package is not a pristine out-of-sample comparison."
        ),
    })
    print(f"\nwrote {out/'final_manifest.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    main(ap.parse_args().config)
