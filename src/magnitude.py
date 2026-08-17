"""Branch A — conditional-mean magnitude, E[|r| | F(T)].

The headline magnitude_score is an L1 statistic, minimised by the conditional
median. The brief asks for the conditional *mean*. We target the mean and accept
the score cost, then report the quantile model as a documented sensitivity.

Every fitted object (imputer, scaler, smearing factor, calibration) is fit on the
fold's training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HARX = ["mag_lag_1", "mag_mean_5", "mag_mean_20", "mag_mean_60", "mag_ewm_20",
        "mag_std_20", "mag_median_20", "rv_full_pct", "rv_final_30_pct", "rv_mean_20",
        "mkt_rv", "mkt_ret_disp", "mkt_on_mean_20", "calendar_gap_days",
        "range_pct", "log_dollar_volume", "bench_gap_abs"]


def _ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, solver="lsqr")),
    ])


def _lgbm(objective: str, cfg: dict, seed: int, **extra) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective=objective, random_state=seed, deterministic=True,
                             force_col_wise=True, verbosity=-1, **cfg, **extra)


def _matrix(frame: pd.DataFrame, cols: list[str], cats: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    x["symbol"] = pd.Categorical(frame["symbol"], categories=cats)
    return x


# --------------------------------------------------------------------------- #
# candidates: each returns a fit(train) -> predict(frame) closure
# --------------------------------------------------------------------------- #
def fit_m0(train, cols, cats, cfg, seed):
    """Trailing 20-session mean magnitude. Also the r2_vs_vol denominator."""
    def predict(frame):
        return frame["mag_mean_20"].to_numpy(float)
    return predict


def fit_m1_harx(train, cols, cats, cfg, seed):
    """log1p target under Ridge, with Duan smearing back to the mean scale."""
    alphas = cfg["ridge_alphas"]
    y = np.log1p(train.actual_magnitude_pct.to_numpy(float))
    best, best_alpha = None, None
    # inner holdout: final 20% of the fold's own training rows, chronological
    cut = train.pred_date.quantile(0.8)
    a_tr, a_va = train[train.pred_date <= cut], train[train.pred_date > cut]
    for alpha in alphas:
        p = _ridge_pipeline(alpha)
        p.fit(a_tr[HARX], np.log1p(a_tr.actual_magnitude_pct))
        sse = np.mean((p.predict(a_va[HARX]) - np.log1p(a_va.actual_magnitude_pct)) ** 2)
        if best is None or sse < best:
            best, best_alpha = sse, alpha
    model = _ridge_pipeline(best_alpha)
    model.fit(train[HARX], y)
    resid = y - model.predict(train[HARX])
    smear = float(np.mean(np.exp(resid)))          # Duan (1983)

    def predict(frame):
        z = model.predict(frame[HARX])
        return np.maximum(np.exp(z) * smear - 1.0, 0.0)
    return predict


# |r| has a point mass at exactly zero (~7.4% of training rows: the T+1
# auction prints at the previous close). Tweedie with 1<p<2 is compound
# Poisson-Gamma and models that mass natively; Gamma needs a positive floor.
GAMMA_FLOOR = 1e-4


def _lgbm_mean_factory(objective, extra_key=None):
    def factory(train, cols, cats, cfg, seed):
        params = dict(cfg["lgbm"])
        extra = {}
        if objective == "tweedie":
            extra["tweedie_variance_power"] = extra_key
        model = _lgbm(objective, params, seed, **extra)
        y = train.actual_magnitude_pct.to_numpy(float)
        if objective == "gamma":
            y = np.maximum(y, GAMMA_FLOOR)
        model.fit(_matrix(train, cols, cats), y,
                  categorical_feature=["symbol"])

        def predict(frame):
            return np.maximum(model.predict(_matrix(frame, cols, cats)), 0.0)
        return predict
    return factory


def fit_m4_logl2(train, cols, cats, cfg, seed):
    """L2 on log1p(|r|) + smearing. Squared loss targets the conditional mean of z."""
    params = dict(cfg["lgbm"])
    model = _lgbm("regression", params, seed)
    y = np.log1p(train.actual_magnitude_pct.to_numpy(float))
    model.fit(_matrix(train, cols, cats), y, categorical_feature=["symbol"])
    resid = y - model.predict(_matrix(train, cols, cats))
    smear = float(np.mean(np.exp(resid)))

    def predict(frame):
        z = model.predict(_matrix(frame, cols, cats))
        return np.maximum(np.exp(z) * smear - 1.0, 0.0)
    return predict


def fit_m6_l2_direct(train, cols, cats, cfg, seed):
    """Squared loss on |r| itself. The most direct conditional-mean estimator:
    no link, no transform, no smearing correction to get wrong."""
    model = _lgbm("regression", dict(cfg["lgbm"]), seed)
    model.fit(_matrix(train, cols, cats), train.actual_magnitude_pct,
              categorical_feature=["symbol"])

    def predict(frame):
        return np.maximum(model.predict(_matrix(frame, cols, cats)), 0.0)
    return predict


def fit_m7_poisson(train, cols, cats, cfg, seed):
    """Poisson log-link: mean-targeting and tolerant of the point mass at zero."""
    model = _lgbm("poisson", dict(cfg["lgbm"]), seed)
    model.fit(_matrix(train, cols, cats), train.actual_magnitude_pct,
              categorical_feature=["symbol"])

    def predict(frame):
        return np.maximum(model.predict(_matrix(frame, cols, cats)), 0.0)
    return predict


def fit_m8_harx_full(train, cols, cats, cfg, seed):
    """log-Ridge on the full feature set rather than the 17 HAR-X columns, + smearing."""
    y = np.log1p(train.actual_magnitude_pct.to_numpy(float))
    cut = train.pred_date.quantile(0.8)
    a_tr, a_va = train[train.pred_date <= cut], train[train.pred_date > cut]
    best, best_alpha = None, cfg["ridge_alphas"][-1]
    for alpha in cfg["ridge_alphas"]:
        p = _ridge_pipeline(alpha)
        p.fit(a_tr[cols], np.log1p(a_tr.actual_magnitude_pct))
        sse = np.mean((p.predict(a_va[cols]) - np.log1p(a_va.actual_magnitude_pct)) ** 2)
        if best is None or sse < best:
            best, best_alpha = sse, alpha
    model = _ridge_pipeline(best_alpha)
    model.fit(train[cols], y)
    smear = float(np.mean(np.exp(y - model.predict(train[cols]))))

    def predict(frame):
        return np.maximum(np.exp(model.predict(frame[cols])) * smear - 1.0, 0.0)
    return predict


def fit_shadow_quantile(train, cols, cats, cfg, seed):
    """Documented sensitivity only: score-optimised, NOT submitted as E[|r|]."""
    params = dict(cfg["lgbm"])
    model = _lgbm("quantile", params, seed, alpha=0.45)
    model.fit(_matrix(train, cols, cats), np.log1p(train.actual_magnitude_pct),
              categorical_feature=["symbol"])

    def predict(frame):
        return np.maximum(np.expm1(model.predict(_matrix(frame, cols, cats))), 0.0)
    return predict


def candidates(cfg) -> dict:
    out = {
        "M0_trailing20": fit_m0,
        "M1_harx_ridge_smear": fit_m1_harx,
        "M2_lgbm_gamma": _lgbm_mean_factory("gamma"),
        "M4_lgbm_logl2_smear": fit_m4_logl2,
        "M6_lgbm_l2_direct": fit_m6_l2_direct,
        "M7_lgbm_poisson": fit_m7_poisson,
        "M8_harx_full_ridge": fit_m8_harx_full,
        "S_quantile45_shadow": fit_shadow_quantile,
    }
    for p in cfg["tweedie_powers"]:
        out[f"M3_lgbm_tweedie_{p}"] = _lgbm_mean_factory("tweedie", p)
    return out


# --------------------------------------------------------------------------- #
def affine_mean_calibration(oof_pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Fit |a| = a0 + b*m on OOF rows, constrained to b >= 0 and non-negative output."""
    ok = np.isfinite(oof_pred) & np.isfinite(actual)
    x, y = oof_pred[ok], actual[ok]
    b, a0 = np.polyfit(x, y, 1)
    return float(a0), float(max(b, 0.0))


def apply_calibration(pred: np.ndarray, a0: float, b: float) -> np.ndarray:
    return np.maximum(a0 + b * pred, 0.0)


def semantic_gate(pred: np.ndarray, actual: np.ndarray, cfg) -> dict:
    """A candidate may only be called E[|r|] if its level is right, not just its ranking."""
    ok = np.isfinite(pred) & np.isfinite(actual)
    p, a = pred[ok], actual[ok]
    mean_gap = 100.0 * (p.mean() / a.mean() - 1.0)
    slope, intercept = np.polyfit(p, a, 1)[0], np.polyfit(p, a, 1)[1]
    g = cfg["mean_gate"]
    return {
        "mean_pred": float(p.mean()),
        "mean_actual": float(a.mean()),
        "mean_gap_pct": float(mean_gap),
        "calib_slope": float(slope),
        "calib_intercept": float(intercept),
        "passes_mean_gate": bool(abs(mean_gap) <= g["max_mean_gap_pct"]),
        "passes_slope_gate": bool(g["slope_low"] <= slope <= g["slope_high"]),
    }
