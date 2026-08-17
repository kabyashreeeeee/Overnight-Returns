"""Stage 3: Branch B (direction) + quantile support model + OOF stores for confidence."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import direction as D
from src import oof as O
from src import metrics as MET
from scipy.stats import spearmanr


def ds(d, a):
    return float(np.sum(d * a) / np.sum(np.abs(a)))


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parent
    out = (root / cfg["paths"]["output_dir"]).resolve()
    seed = int(cfg["seed"])

    panel = pd.read_parquet(out / "panel.parquet")
    cols = pd.read_csv(out / "feature_columns.csv")["feature"].tolist()
    fd = pd.read_csv(out / "oof_folds.csv", parse_dates=["train_end", "predict_start", "predict_end"])
    folds = [O.Fold(int(r.fold), r.train_end, r.predict_start, r.predict_end, []) for r in fd.itertuples()]

    train = panel[panel.split == "train"].copy().reset_index(drop=True)
    valid = panel[panel.split == "valid"].copy().reset_index(drop=True)
    cats = sorted(panel.symbol.unique())
    for f in (train, valid):
        f["resid_target"] = f.actual_return_pct - f.groupby("pred_date").actual_return_pct.transform("mean")

    # ---------------- OOF: market, residual, quantiles ----------------
    n = len(train)
    oof_mkt = np.full(n, np.nan)
    oof_res = np.full(n, np.nan)
    oof_q = np.full((n, len(cfg["quantiles"]["levels"])), np.nan)
    cross_rates = []
    for f in folds:
        t0 = time.time()
        tr = (train.pred_date <= f.train_end).to_numpy()
        pr = train.pred_date.between(f.predict_start, f.predict_end).to_numpy()
        mtr = D.market_frame(train[tr])
        mmodel, alpha = D.fit_market(mtr, cfg["direction"]["market_ridge_alphas"], seed)
        mpred_frame = D.market_frame(train[pr])
        mp = pd.Series(mmodel.predict(mpred_frame[D.MARKET_FEATURES]), index=mpred_frame.pred_date)
        oof_mkt[pr] = train.loc[pr, "pred_date"].map(mp).to_numpy()

        rfn = D.fit_residual(train[tr], cols, cats, cfg["direction"], seed)
        oof_res[pr] = rfn(train[pr])

        qfn = D.fit_quantiles(train[tr], cols, cats, cfg["quantiles"], seed)
        q, cr = qfn(train[pr])
        oof_q[pr] = q
        cross_rates.append(cr)
        print(f"  fold {f.index}: market alpha={alpha:>7.0f}  rows={pr.sum():6d}  "
              f"quantile crossings pre-fix {cr:5.2%}  {time.time()-t0:6.1f}s", flush=True)

    have = np.isfinite(oof_mkt) & np.isfinite(oof_res)
    if not np.isfinite(oof_q[have]).all():
        raise AssertionError("quantile model emitted non-finite OOF predictions")
    print(f"\nOOF coverage {have.mean():.1%}; mean quantile crossing rate {np.mean(cross_rates):.2%}")

    # ---------------- select residual shrinkage lambda on OOF ----------------
    a_oof = train.actual_return_pct.to_numpy()[have]
    e_oof = train.resid_target.to_numpy()[have]
    rows = []
    for lam in cfg["direction"]["shrinkage_grid"]:
        mu = oof_mkt[have] + lam * oof_res[have]
        d = np.where(mu >= 0, 1.0, -1.0)
        rows.append({"lambda": lam, "pooled_ds": ds(d, a_oof), "residual_ds": ds(d, e_oof),
                     "hit": float(np.mean(d == np.where(a_oof >= 0, 1, -1))),
                     "frac_long": float(np.mean(d > 0))})
    lam_tab = pd.DataFrame(rows)
    lam_tab["objective"] = lam_tab.pooled_ds          # declared: maximise pooled, tie-break residual
    best = lam_tab.sort_values(["objective", "residual_ds"], ascending=False).iloc[0]
    lam_star = float(best["lambda"])
    lam_tab.to_csv(out / "direction_lambda_oof.csv", index=False)
    print("\nresidual shrinkage selected on INNER OOF (never validation):")
    print(lam_tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  lambda* = {lam_star}")

    # ---------------- refit on full train, one look at validation ----------------
    mtr = D.market_frame(train)
    mmodel, alpha = D.fit_market(mtr, cfg["direction"]["market_ridge_alphas"], seed)
    mval = D.market_frame(valid)
    mp = pd.Series(mmodel.predict(mval[D.MARKET_FEATURES]), index=mval.pred_date)
    v_mkt = valid.pred_date.map(mp).to_numpy()
    rfn = D.fit_residual(train, cols, cats, cfg["direction"], seed)
    v_res = rfn(valid)
    qfn = D.fit_quantiles(train, cols, cats, cfg["quantiles"], seed)
    v_q, v_cross = qfn(valid)
    if not (np.isfinite(v_mkt).all() and np.isfinite(v_res).all()
            and np.isfinite(v_q).all()):
        raise AssertionError("direction stage emitted non-finite validation predictions")

    av, ev = valid.actual_return_pct.to_numpy(), valid.resid_target.to_numpy()
    print(f"\n=== OFFICIAL VALIDATION (one look). quantile crossings pre-fix {v_cross:.2%} ===")
    print(f"{'rule':34s} {'pooled':>8s} {'residual':>9s} {'hit':>7s} {'frac_long':>10s}")
    for lab, mu in [("market only", v_mkt),
                    ("residual only", v_res),
                    (f"market + {lam_star} x residual", v_mkt + lam_star * v_res)]:
        d = np.where(mu >= 0, 1.0, -1.0)
        print(f"{lab:34s} {ds(d, av):8.4f} {ds(d, ev):9.4f} "
              f"{np.mean(d == np.where(av >= 0, 1, -1)):7.4f} {np.mean(d > 0):10.4f}")
    d = np.where(np.ones(len(valid)) >= 0, 1.0, -1.0)
    print(f"{'always +1 (naive)':34s} {ds(d, av):8.4f} {ds(d, ev):9.4f} "
          f"{np.mean(d == np.where(av >= 0, 1, -1)):7.4f} {1.0:10.4f}")

    # ---------------- persist stores for the confidence branches ----------------
    levels = cfg["quantiles"]["levels"]
    tr_store = train[["pred_date", "symbol", "actual_return_pct", "resid_target",
                      "actual_magnitude_pct"]].copy()
    tr_store["mu_market"] = oof_mkt
    tr_store["mu_residual"] = oof_res
    tr_store["mu_total"] = oof_mkt + lam_star * oof_res
    tr_store["p_up"] = np.nan
    ok = np.isfinite(oof_q).all(axis=1)
    tr_store.loc[ok, "p_up"] = D.prob_up_from_quantiles(oof_q[ok], levels)
    dfeat = pd.DataFrame(np.nan, index=tr_store.index,
                         columns=D.distribution_features(oof_q[:1], levels).columns)
    dfeat.loc[ok, :] = D.distribution_features(oof_q[ok], levels).to_numpy()
    tr_store = pd.concat([tr_store, dfeat], axis=1)
    tr_store.to_parquet(out / "direction_oof.parquet", index=False)

    va_store = valid[["pred_date", "symbol", "actual_return_pct", "resid_target",
                      "actual_magnitude_pct"]].copy()
    va_store["mu_market"] = v_mkt
    va_store["mu_residual"] = v_res
    va_store["mu_total"] = v_mkt + lam_star * v_res
    va_store["p_up"] = D.prob_up_from_quantiles(v_q, levels)
    va_store = pd.concat([va_store, D.distribution_features(v_q, levels)], axis=1)
    va_store.to_parquet(out / "direction_valid.parquet", index=False)
    pd.Series({"lambda": lam_star, "market_alpha": alpha}).to_csv(out / "direction_selection.csv")

    # Post-final research diagnostic: isolate the official-close mechanical basis.
    # This never participates in model selection and is evaluated on validation only.
    no_benchmark_cols = [c for c in cols if "bench_gap" not in c]
    no_benchmark_market = [c for c in D.MARKET_FEATURES if c != "mkt_bench_gap"]
    mtr_nb = D.market_frame(train, no_benchmark_market)
    mm_nb, alpha_nb = D.fit_market(
        mtr_nb, cfg["direction"]["market_ridge_alphas"], seed, no_benchmark_market
    )
    mval_nb = D.market_frame(valid, no_benchmark_market)
    mp_nb = pd.Series(
        mm_nb.predict(mval_nb[no_benchmark_market]), index=mval_nb.pred_date
    )
    v_mkt_nb = valid.pred_date.map(mp_nb).to_numpy()
    v_res_nb = D.fit_residual(
        train, no_benchmark_cols, cats, cfg["direction"], seed
    )(valid)

    def diagnostic_row(name: str, score: np.ndarray) -> dict:
        direction = np.where(score >= 0, 1.0, -1.0)
        by = pd.DataFrame({
            "symbol": valid.symbol.to_numpy(),
            "correct": direction == np.where(av >= 0, 1.0, -1.0),
            "naive": av >= 0,
        }).groupby("symbol").agg(n=("correct", "size"), hit=("correct", "mean"), naive=("naive", "mean"))
        by = by[by.n >= 20]
        daily_ic = []
        temp = pd.DataFrame({"date": valid.pred_date, "score": score, "resid": ev})
        for _, group in temp.groupby("date"):
            if group.score.nunique() > 1 and group.resid.nunique() > 1:
                daily_ic.append(spearmanr(group.score, group.resid).statistic)
        return {
            "model": name,
            "pooled_direction_score": ds(direction, av),
            "residual_direction_score": ds(direction, ev),
            "hit_rate": float(np.mean(direction == np.where(av >= 0, 1.0, -1.0))),
            "frac_stocks_beat_naive": float((by.hit > by.naive).mean()),
            "residual_rank_ic": float(np.nanmean(daily_ic)),
            "fraction_long": float(np.mean(direction > 0)),
        }

    full_score = v_mkt + lam_star * v_res
    ablated_score = v_mkt_nb + lam_star * v_res_nb
    ablation = pd.DataFrame([
        diagnostic_row("full_assignment_features", full_score),
        diagnostic_row("economic_sensitivity_no_benchmark_basis", ablated_score),
    ])
    for scope, target in [("pooled", av), ("residual", ev)]:
        mean, lo, hi = MET.block_bootstrap_diff(
            valid,
            np.where(full_score >= 0, 1.0, -1.0),
            np.where(ablated_score >= 0, 1.0, -1.0),
            target,
            block_sessions=int(cfg["bootstrap"]["block_sessions"]),
            draws=int(cfg["bootstrap"]["draws"]),
            seed=seed,
        )
        ablation[f"full_minus_ablated_{scope}_mean"] = mean
        ablation[f"full_minus_ablated_{scope}_ci_low"] = lo
        ablation[f"full_minus_ablated_{scope}_ci_high"] = hi
    ablation["market_alpha"] = [alpha, alpha_nb]
    ablation.to_csv(out / "direction_benchmark_ablation.csv", index=False)
    print(f"\nwrote direction_oof.parquet ({ok.sum()} usable OOF rows) and direction_valid.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    main(ap.parse_args().config)
