"""Branch B — expected-return direction, plus the quantile distribution support model.

r_i,T = mu^M_T + mu^R_i,T + eps. The market leg has ~935 training observations and
gets a heavily regularised linear model; the residual leg has ~180k and can carry a
tree. Shrinkage lambda on the residual leg is a signal-strength parameter selected
on inner OOF, not a post-hoc demeaning of the emitted score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MARKET_FEATURES = [
    "mkt_ret_oc", "mkt_ret_cc", "mkt_rv", "mkt_rv_disp", "mkt_ret_disp", "mkt_final60",
    "mkt_breadth_up", "mkt_near_high", "mkt_bench_gap",
    "mkt_on_lag_1", "mkt_on_lag_2", "mkt_on_mean_20", "mkt_on_mean_60",
    "mkt_on_pos_share_20", "mkt_on_std_20", "calendar_gap_days", "weekday", "is_long_gap",
]


def _matrix(frame, cols, cats):
    x = frame[cols].copy()
    x["symbol"] = pd.Categorical(frame["symbol"], categories=cats)
    return x


def market_frame(panel: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """One row per session: the cross-sectional mean overnight return and market features."""
    features = MARKET_FEATURES if features is None else features
    agg = {c: "first" for c in features}
    agg["actual_return_pct"] = "mean"
    m = panel.groupby("pred_date", as_index=False).agg(agg)
    return m.rename(columns={"actual_return_pct": "mkt_target"}).sort_values("pred_date")


def fit_market(
    train_market: pd.DataFrame,
    alphas: list[float],
    seed: int,
    features: list[str] | None = None,
):
    """Chronological inner holdout picks alpha; ~935 rows means heavy shrinkage."""
    features = MARKET_FEATURES if features is None else features
    cut = train_market.pred_date.quantile(0.8)
    a_tr = train_market[train_market.pred_date <= cut]
    a_va = train_market[train_market.pred_date > cut]
    best, best_alpha = None, alphas[-1]
    for alpha in alphas:
        p = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        p.fit(a_tr[features], a_tr.mkt_target)
        sse = np.mean((p.predict(a_va[features]) - a_va.mkt_target) ** 2)
        if best is None or sse < best:
            best, best_alpha = sse, alpha
    model = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scale", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
    model.fit(train_market[features], train_market.mkt_target)
    return model, best_alpha


def fit_residual(train: pd.DataFrame, cols: list[str], cats: list[str], cfg: dict, seed: int):
    """Squared loss on the cross-sectionally demeaned target -> conditional mean of e."""
    model = lgb.LGBMRegressor(objective="regression", random_state=seed, deterministic=True,
                              force_col_wise=True, verbosity=-1, **cfg["residual_lgbm"])
    target = train.actual_return_pct - train.groupby("pred_date").actual_return_pct.transform("mean")
    model.fit(_matrix(train, cols, cats), target, categorical_feature=["symbol"])

    def predict(frame):
        return model.predict(_matrix(frame, cols, cats))
    return predict


def fit_quantiles(train: pd.DataFrame, cols: list[str], cats: list[str], cfg: dict, seed: int):
    """Conditional distribution of the total signed return. Support model only:
    supplies P(r>0), interval width and tail mass to the confidence branch."""
    models = {}
    for tau in cfg["levels"]:
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau, random_state=seed,
                              deterministic=True, force_col_wise=True, verbosity=-1,
                              **cfg["lgbm"])
        m.fit(_matrix(train, cols, cats), train.actual_return_pct,
              categorical_feature=["symbol"])
        models[tau] = m

    def predict(frame):
        x = _matrix(frame, cols, cats)
        raw = np.column_stack([models[t].predict(x) for t in cfg["levels"]])
        crossings = float(np.mean(np.diff(raw, axis=1) < 0))
        q = np.sort(raw, axis=1)          # monotone rearrangement
        return q, crossings
    return predict


def prob_up_from_quantiles(q: np.ndarray, levels: list[float]) -> np.ndarray:
    """Interpolate the conditional CDF at zero; return P(r > 0)."""
    tau = np.asarray(levels, float)
    n = q.shape[0]
    f0 = np.empty(n)
    for i in range(n):
        row = q[i]
        if row[0] > 0:
            f0[i] = 0.0
        elif row[-1] <= 0:
            f0[i] = 1.0
        else:
            j = np.searchsorted(row, 0.0, side="right") - 1
            j = min(max(j, 0), len(row) - 2)
            lo, hi = row[j], row[j + 1]
            w = 0.0 if hi == lo else (0.0 - lo) / (hi - lo)
            f0[i] = tau[j] + w * (tau[j + 1] - tau[j])
    return np.clip(1.0 - f0, 1e-6, 1 - 1e-6)


def distribution_features(q: np.ndarray, levels: list[float]) -> pd.DataFrame:
    tau = np.asarray(levels, float)
    med = q[:, np.argmin(np.abs(tau - 0.5))]
    lo = q[:, np.argmin(np.abs(tau - 0.10))]
    hi = q[:, np.argmin(np.abs(tau - 0.90))]
    lo5 = q[:, np.argmin(np.abs(tau - 0.05))]
    hi5 = q[:, np.argmin(np.abs(tau - 0.95))]
    return pd.DataFrame({
        "q_median": med,
        "q_iqr80": hi - lo,
        "q_iqr90": hi5 - lo5,
        "q_skew": (hi + lo - 2 * med) / np.maximum(hi - lo, 1e-9),
        "q_left_tail": np.abs(lo5),
        "q_right_tail": np.abs(hi5),
        "q_median_abs": np.abs(med),
    })
