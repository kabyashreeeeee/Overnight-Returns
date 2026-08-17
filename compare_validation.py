"""Head-to-head validation comparison with the earlier submission."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import metrics as MET

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OLD = ROOT.parent / "submission" / "existing_submission"

fresh = pd.read_parquet(OUT / "fresh_validation_predictions.parquet")
fresh["pred_date"] = pd.to_datetime(fresh.pred_date)
panel = pd.read_parquet(OUT / "panel.parquet")
fresh_history = MET.add_vol_baseline(
    panel[["pred_date", "symbol", "actual_magnitude_pct"]].copy()
)
fresh = fresh.merge(
    fresh_history[["pred_date", "symbol", "vol_baseline_20"]],
    on=["pred_date", "symbol"], how="left", validate="1:1",
)

op = pd.read_csv(OLD / "predictions.csv", parse_dates=["pred_date"])
oa = pd.read_csv(OLD / "actuals.csv", parse_dates=["pred_date"])
old = MET.add_vol_baseline(op.merge(oa, on=["pred_date", "symbol"]))
old = old[old.split == "valid"].copy()

key = ["pred_date", "symbol"]
common = fresh[key].merge(old[key], on=key)
fresh = fresh.merge(common, on=key).sort_values(key).reset_index(drop=True)
old = old.merge(common, on=key).sort_values(key).reset_index(drop=True)
print(f"comparing on {len(common)} identical validation rows "
      f"({fresh.pred_date.nunique()} sessions, {fresh.symbol.nunique()} symbols)\n")

for f in (fresh, old):
    f["split"] = "valid"
# Residual comparison must use one common-row universe mean, not each system's
# slightly different coverage denominator.
shared_mean = fresh.groupby("pred_date").actual_return_pct.transform("mean")
fresh["universe_mean_pct"] = shared_mean.to_numpy()
old["actual_return_pct"] = fresh.actual_return_pct.to_numpy()
old["actual_magnitude_pct"] = fresh.actual_magnitude_pct.to_numpy()
old["universe_mean_pct"] = shared_mean.to_numpy()

rows = []
for label, frame in [("fresh", fresh), ("existing", old)]:
    for scope in ["pooled", "residual"]:
        for m, (v, n) in MET.compute_scope(frame, scope).items():
            rows.append({"system": label, "scope": scope, "metric": m, "value": v})
tab = pd.DataFrame(rows).pivot_table(index=["scope", "metric"], columns="system", values="value")
tab["delta"] = tab["fresh"] - tab["existing"]

order = ["direction_score", "directional_return_pct", "hit_rate", "magnitude_score",
         "mae", "rmse", "rank_ic", "rank_ic_t", "r2_vs_vol",
         "conf_direction_score", "conf_direction_lift", "brier", "brier_skill",
         "log_loss", "ece_10", "conf_magnitude_score", "conf_mag_gradient",
         "mae_conf_top_decile", "mae_conf_bottom_decile",
         "frac_stocks_hit_gt_50", "frac_stocks_beat_naive", "var_share_universe"]
pd.set_option("display.width", 200)
for scope in ["pooled", "residual"]:
    sub = tab.loc[scope].reindex([m for m in order if (scope, m) in tab.index])
    print(f"=== {scope.upper()} (validation) ===")
    print(sub[["existing", "fresh", "delta"]].to_string(float_format=lambda v: f"{v:9.4f}"))
    print()

# estimand check: is pred_magnitude actually E[|r|]?
print("=== ESTIMAND CHECK: does pred_magnitude behave like E[|r|]? ===")
for label, frame in [("fresh", fresh), ("existing", old)]:
    p = frame.pred_magnitude_pct.to_numpy()
    a = frame.actual_magnitude_pct.to_numpy()
    b, a0 = np.polyfit(p, a, 1)
    print(f"  {label:9s} mean pred {p.mean():.4f} vs mean |r| {a.mean():.4f}  "
          f"gap {100*(p.mean()/a.mean()-1):+6.2f}%   |a| = {a0:+.4f} + {b:.4f}*m   "
          f"P(|a|>m) = {100*(a>p).mean():.1f}%")

# non-degeneracy of conf_magnitude
from scipy.stats import spearmanr
print("\n=== conf_magnitude: reliability or just inverse scale? ===")
for label, frame in [("fresh", fresh), ("existing", old)]:
    err = (frame.pred_magnitude_pct - frame.actual_magnitude_pct).abs().to_numpy()
    c = frame.conf_magnitude.to_numpy()
    m = frame.pred_magnitude_pct.to_numpy()
    dec = pd.qcut(pd.Series(m).rank(method="first"), 10, labels=False).to_numpy()
    within = np.mean([spearmanr(c[dec == k], -err[dec == k]).statistic for k in range(10)])
    base = spearmanr(-m, -err).statistic
    print(f"  {label:9s} score {spearmanr(c,-err).statistic:+.4f} | "
          f"vs -pred_magnitude {base:+.4f} (incremental {spearmanr(c,-err).statistic-base:+.4f}) | "
          f"within-decile {within:+.4f} | corr(conf,mag) {spearmanr(c,m).statistic:+.3f}")

# block bootstrap on the direction difference
print("\n=== block bootstrap: fresh vs existing direction_score (5-session blocks) ===")
for scope, tgt in [("pooled", fresh.actual_return_pct.to_numpy()),
                   ("residual", (fresh.actual_return_pct - fresh.universe_mean_pct).to_numpy())]:
    mean, lo, hi = MET.block_bootstrap_diff(
        fresh, fresh.pred_direction.to_numpy().astype(float),
        old.pred_direction.to_numpy().astype(float), tgt, draws=1000)
    print(f"  {scope:9s} delta {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
