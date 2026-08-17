"""Branch C (direction confidence) and Branch D (magnitude reliability).

Branch C estimates P(emitted direction is realised) from a *different* object than
the direction score: the quantile model's conditional CDF at zero, calibrated on
genuine OOF correctness labels.

Branch D deliberately splits reliability into a scale term and a forecast-specific
term, because raw |m - |a|| is dominated by aleatoric variance and any model of it
collapses onto -pred_magnitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CONF_DIR_FEATURES = [
    "p_up_emitted", "abs_mu", "mu_over_mag", "mu_market", "mu_residual",
    "market_residual_agree", "residual_flips_market", "q_iqr80", "q_iqr90",
    "q_skew", "q_median_abs", "q_left_tail", "q_right_tail",
    "mkt_rv", "mkt_ret_disp", "calendar_gap_days", "minute_coverage",
    "history_count", "bench_gap_abs",
]

CONF_MAG_FEATURES = [
    "pred_magnitude_pct", "q_iqr80", "q_iqr90", "q_median_abs",
    "mag_mean_20", "mag_std_20", "rv_full_pct", "rv_z_60", "rv_of_rv_20",
    "mkt_rv", "mkt_ret_disp", "calendar_gap_days", "minute_coverage",
    "history_count", "n_missing_features", "bench_gap_abs", "log_dollar_volume",
    "err_hist_mean_20", "err_hist_std_20",
]


def _pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
        ("scale", StandardScaler()),
        ("est", estimator),
    ])


# --------------------------------------------------------------------------- #
# Branch C
# --------------------------------------------------------------------------- #
def build_direction_confidence_inputs(store: pd.DataFrame, panel: pd.DataFrame,
                                      pred_mag: np.ndarray) -> pd.DataFrame:
    f = store.copy()
    f["pred_magnitude_pct"] = pred_mag
    f["base_direction"] = np.where(f.mu_total >= 0, 1.0, -1.0)
    f["p_up_emitted"] = np.where(f.base_direction > 0, f.p_up, 1.0 - f.p_up)
    f["abs_mu"] = f.mu_total.abs()
    f["mu_over_mag"] = f.mu_total.abs() / np.maximum(f.pred_magnitude_pct, 1e-6)
    f["market_residual_agree"] = (np.sign(f.mu_market) == np.sign(f.mu_residual)).astype(float)
    f["residual_flips_market"] = (np.sign(f.mu_market) != np.sign(f.mu_total)).astype(float)
    extra = panel[["pred_date", "symbol", "mkt_rv", "mkt_ret_disp", "calendar_gap_days",
                   "minute_coverage", "history_count", "bench_gap_abs"]]
    f = f.merge(extra, on=["pred_date", "symbol"], how="left")
    f["correct"] = (f.base_direction == np.where(f.actual_return_pct >= 0, 1.0, -1.0)).astype(float)
    return f


def direction_confidence_candidates(seed: int) -> dict:
    """C5/C6 added after the first validation pass: the initial registry fed the
    calibrator only the quantile model's P(r>0), which is misaligned with an emitted
    direction taken from sign(E[r]) — the two disagree on ~13% of rows under skew.
    C5/C6 also supply the aligned mean margin. This is a disclosed second look at
    validation, not a fresh single-look claim."""
    return {
        "C0_constant": "constant",
        "C1_raw_quantile_prob": "raw",
        "C2_platt": _pipeline(LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        "C3_isotonic": "isotonic",
        "C4_logistic_meta": _pipeline(LogisticRegression(C=0.1, max_iter=2000, random_state=seed)),
        "C5_isotonic_margin": "isotonic_margin",
        "C6_logistic_margin_prob": _pipeline(
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
    }


# inputs for C6: the aligned margin and the distribution probability are fitted
# independently, so combining them is not a rescaling of a single score
C6_FEATURES = ["abs_mu", "p_up_emitted", "mu_over_mag", "q_iqr80", "q_skew",
               "market_residual_agree", "mkt_ret_disp", "calendar_gap_days"]


def fit_direction_confidence(name, model, train: pd.DataFrame, seed: int):
    y = train.correct.to_numpy(float)
    if name == "C0_constant":
        rate = float(y.mean())
        return lambda frame: np.full(len(frame), rate)
    if name == "C1_raw_quantile_prob":
        return lambda frame: np.clip(frame.p_up_emitted.to_numpy(float), 1e-6, 1 - 1e-6)
    if name == "C3_isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(train.p_up_emitted.to_numpy(float), y)
        return lambda frame: np.clip(iso.predict(frame.p_up_emitted.to_numpy(float)), 1e-6, 1 - 1e-6)
    if name == "C5_isotonic_margin":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(train.abs_mu.to_numpy(float), y)
        return lambda frame: np.clip(iso.predict(frame.abs_mu.to_numpy(float)), 1e-6, 1 - 1e-6)
    if name == "C2_platt":
        model.fit(train[["p_up_emitted"]], y)
        return lambda frame: np.clip(model.predict_proba(frame[["p_up_emitted"]])[:, 1], 1e-6, 1 - 1e-6)
    if name == "C6_logistic_margin_prob":
        model.fit(train[C6_FEATURES], y)
        return lambda frame: np.clip(model.predict_proba(frame[C6_FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    model.fit(train[CONF_DIR_FEATURES], y)
    return lambda frame: np.clip(model.predict_proba(frame[CONF_DIR_FEATURES])[:, 1], 1e-6, 1 - 1e-6)


def emit(base_direction: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Declared before validation: if P(correct) < 0.5, flip and report 1-q."""
    flip = q < 0.5
    d = np.where(flip, -base_direction, base_direction)
    conf = np.where(flip, 1.0 - q, q)
    return d, conf


def calibration_report(q: np.ndarray, correct: np.ndarray) -> dict:
    brier = float(np.mean((q - correct) ** 2))
    ref = float(np.mean((correct.mean() - correct) ** 2))
    qc = np.clip(q, 1e-6, 1 - 1e-6)
    bucket = np.clip(np.digitize(q, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    ece = 0.0
    for k in range(10):
        m = bucket == k
        if m.sum():
            ece += m.sum() / len(q) * abs(correct[m].mean() - q[m].mean())
    return {"brier": brier, "brier_skill": 1 - brier / ref if ref else np.nan,
            "log_loss": float(-np.mean(correct * np.log(qc) + (1 - correct) * np.log(1 - qc))),
            "ece_10": float(ece), "mean_conf": float(q.mean()), "acc": float(correct.mean())}


# --------------------------------------------------------------------------- #
# Branch D
# --------------------------------------------------------------------------- #
def add_error_history(frame: pd.DataFrame, err_col: str = "oof_abs_error") -> pd.DataFrame:
    """Strictly lagged per-symbol history of the model's own realised errors."""
    f = frame.sort_values(["symbol", "pred_date"]).copy()
    g = f.groupby("symbol", sort=False)[err_col]
    f["err_hist_mean_20"] = g.transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    f["err_hist_std_20"] = g.transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    return f


def attach_error_history(frame: pd.DataFrame, oof_history: pd.DataFrame) -> pd.DataFrame:
    """Attach error history without moving future training information backwards.

    Training rows receive the history available on that exact date. Validation
    and test rows may use the last history learned in training, because it is
    already observable by then. Rows before OOF coverage remain missing and are
    handled by the model's training-only imputer.
    """
    keys = ["pred_date", "symbol"]
    cols = ["err_hist_mean_20", "err_hist_std_20"]
    missing = [c for c in keys + cols if c not in oof_history]
    if missing:
        raise KeyError(f"OOF error history is missing columns: {missing}")

    out = frame.copy()
    history = (
        oof_history[keys + cols]
        .sort_values(keys)
        .drop_duplicates(keys, keep="last")
    )
    lookup = history.set_index(keys)
    row_index = pd.MultiIndex.from_frame(out[keys])
    for col in cols:
        out[col] = lookup[col].reindex(row_index).to_numpy()

    last = history.groupby("symbol", sort=False)[cols].last()
    after_train = ~out["split"].eq("train")
    for col in cols:
        out.loc[after_train, col] = out.loc[after_train, "symbol"].map(last[col])
    return out


def fit_two_stage_reliability(train: pd.DataFrame, seed: int, use_tree: bool = False):
    """Stage 1: isotonic b(m)=E[err|m]. Stage 2: model the residual error u=err-b(m)."""
    m = train.pred_magnitude_pct.to_numpy(float)
    err = train.oof_abs_error.to_numpy(float)
    scale = IsotonicRegression(out_of_bounds="clip")
    scale.fit(m, err)
    u = err - scale.predict(m)

    if use_tree:
        est = lgb.LGBMRegressor(objective="regression", num_leaves=7, min_child_samples=500,
                                n_estimators=300, learning_rate=0.03, reg_lambda=10.0,
                                random_state=seed, deterministic=True, force_col_wise=True,
                                verbosity=-1, n_jobs=8)
        est.fit(train[CONF_MAG_FEATURES], u)
        stage2 = lambda frame: est.predict(frame[CONF_MAG_FEATURES])
    else:
        pipe = _pipeline(Ridge(alpha=100.0))
        pipe.fit(train[CONF_MAG_FEATURES], u)
        stage2 = lambda frame: pipe.predict(frame[CONF_MAG_FEATURES])

    def predict_error(frame):
        return scale.predict(frame.pred_magnitude_pct.to_numpy(float)) + stage2(frame)
    return predict_error, scale


def risk_to_confidence(pred_err: np.ndarray, train_pred_err: np.ndarray) -> np.ndarray:
    """Map predicted error to [0,1] through a TRAIN-ONLY empirical CDF; low error -> high conf."""
    ranks = np.searchsorted(np.sort(train_pred_err), pred_err, side="left") / len(train_pred_err)
    return np.clip(1.0 - ranks, 0.0, 1.0)


def reliability_report(conf: np.ndarray, err: np.ndarray, mag: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    dec = pd.qcut(pd.Series(mag).rank(method="first"), 10, labels=False).to_numpy()
    within = [spearmanr(conf[dec == k], -err[dec == k]).statistic for k in range(10)]
    within = [w for w in within if np.isfinite(w)]
    cdec = pd.qcut(pd.Series(conf).rank(method="first"), 10, labels=False).to_numpy()
    return {
        "conf_magnitude_score": float(spearmanr(conf, -err).statistic),
        "within_magnitude_decile": float(np.mean(within)),
        "corr_with_magnitude": float(spearmanr(conf, mag).statistic),
        "mae_top_decile": float(err[cdec == 9].mean()),
        "mae_bottom_decile": float(err[cdec == 0].mean()),
        "conf_mag_gradient": float(err[cdec == 0].mean() - err[cdec == 9].mean()),
    }
