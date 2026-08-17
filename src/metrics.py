"""Assignment metrics, implemented directly from Section 3.3 of the brief."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

POOLED_METRICS = [
    "direction_score", "directional_return_pct", "magnitude_score", "conf_direction_score",
    "conf_direction_lift", "conf_magnitude_score", "hit_rate", "precision_up", "recall_up",
    "f1_up", "brier", "brier_skill", "log_loss", "ece_10", "mae", "rmse", "rank_ic",
    "rank_ic_t", "r2_vs_vol", "mae_conf_top_decile", "mae_conf_bottom_decile",
    "conf_mag_gradient", "frac_stocks_hit_gt_50", "frac_stocks_beat_naive", "var_share_universe",
]
RESIDUAL_METRICS = [
    "direction_score", "directional_return_pct", "magnitude_score", "conf_direction_score",
    "conf_direction_lift", "conf_magnitude_score", "hit_rate", "precision_up", "recall_up",
    "f1_up", "brier", "brier_skill", "log_loss", "ece_10", "rank_ic", "rank_ic_t",
]


def _safe_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    return float(spearmanr(a, b).statistic)


def _daily_rank_ic(dates, pred, actual):
    frame = pd.DataFrame({"d": dates, "p": pred, "a": actual})
    vals = []
    for _, g in frame.groupby("d", sort=True):
        r = _safe_spearman(g.p, g.a)
        if np.isfinite(r):
            vals.append(r)
    v = np.asarray(vals)
    if len(v) < 2:
        return np.nan, np.nan, len(v)
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if v.std(ddof=1) > 0 else np.nan
    return float(v.mean()), float(t), len(v)


def compute_scope(frame: pd.DataFrame, scope: str) -> dict[str, tuple[float, int]]:
    """frame needs: pred_date, symbol, pred_magnitude_pct, pred_direction,
    conf_direction, conf_magnitude, actual_return_pct, universe_mean_pct,
    and (pooled only) vol_baseline_20."""
    a = frame.actual_return_pct.to_numpy(float)
    if scope == "residual":
        a = a - frame.universe_mean_pct.to_numpy(float)
    m = frame.pred_magnitude_pct.to_numpy(float)
    d = frame.pred_direction.to_numpy(float)
    p = frame.conf_direction.to_numpy(float)
    c = frame.conf_magnitude.to_numpy(float)

    absa = np.abs(a)
    correct = (np.where(a >= 0, 1.0, -1.0) == d).astype(float)
    n = len(frame)
    out: dict[str, tuple[float, int]] = {}

    out["direction_score"] = (float(np.sum(d * a) / np.sum(absa)), n)
    out["directional_return_pct"] = (float(np.mean(d * a)), n)
    err = np.abs(m - absa)
    out["magnitude_score"] = (float(1 - err.sum() / absa.sum()), n)
    w = 2 * p - 1
    cds = float(np.sum(w * d * a) / np.sum(w * absa))
    out["conf_direction_score"] = (cds, n)
    out["conf_direction_lift"] = (cds - out["direction_score"][0], n)
    out["conf_magnitude_score"] = (_safe_spearman(c, -err), n)
    out["hit_rate"] = (float(correct.mean()), n)

    up, a_up = d == 1, a >= 0
    prec = float(a_up[up].mean()) if up.sum() else np.nan
    rec = float(up[a_up].mean()) if a_up.sum() else np.nan
    out["precision_up"] = (prec, int(up.sum()))
    out["recall_up"] = (rec, int(a_up.sum()))
    out["f1_up"] = (float(2 * prec * rec / (prec + rec)) if (prec + rec) else np.nan, n)

    brier = float(np.mean((p - correct) ** 2))
    out["brier"] = (brier, n)
    ref = float(np.mean((correct.mean() - correct) ** 2))
    out["brier_skill"] = (float(1 - brier / ref) if ref else np.nan, n)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out["log_loss"] = (float(-np.mean(correct * np.log(pc) + (1 - correct) * np.log(1 - pc))), n)

    bucket = np.clip(np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=False), 0, 9)
    ece = 0.0
    for k in range(10):
        msk = bucket == k
        if msk.sum():
            ece += msk.sum() / n * abs(correct[msk].mean() - p[msk].mean())
    out["ece_10"] = (float(ece), n)

    ic, ic_t, n_days = _daily_rank_ic(frame.pred_date.to_numpy(), m, absa)
    out["rank_ic"] = (ic, n_days)
    out["rank_ic_t"] = (ic_t, n_days)

    if scope != "pooled":
        return {k: out[k] for k in RESIDUAL_METRICS}

    out["mae"] = (float(err.mean()), n)
    out["rmse"] = (float(np.sqrt(np.mean((m - absa) ** 2))), n)

    if "vol_baseline_20" in frame:
        v = frame.vol_baseline_20.to_numpy(float)
        fin = np.isfinite(v)
        sse_ref = float(np.sum((v[fin] - absa[fin]) ** 2))
        out["r2_vs_vol"] = (
            float(1 - np.sum((m[fin] - absa[fin]) ** 2) / sse_ref) if sse_ref else np.nan,
            int(fin.sum()))
    else:
        out["r2_vs_vol"] = (np.nan, 0)

    if np.unique(c).size < 2:
        top = bottom = np.ones(n, dtype=bool)
    else:
        dec = pd.qcut(pd.Series(c).rank(method="first"), 10, labels=False).to_numpy()
        top, bottom = dec == 9, dec == 0
    mt, mb = float(err[top].mean()), float(err[bottom].mean())
    out["mae_conf_top_decile"] = (mt, int(top.sum()))
    out["mae_conf_bottom_decile"] = (mb, int(bottom.sum()))
    out["conf_mag_gradient"] = (mb - mt, n)

    by = pd.DataFrame({"symbol": frame.symbol.to_numpy(), "correct": correct,
                       "up": a_up.astype(float)}).groupby("symbol").agg(
        n=("correct", "size"), hit=("correct", "mean"), naive=("up", "mean"))
    by = by[by.n >= 20]
    out["frac_stocks_hit_gt_50"] = (float((by.hit > 0.5).mean()) if len(by) else np.nan, len(by))
    out["frac_stocks_beat_naive"] = (float((by.hit > by.naive).mean()) if len(by) else np.nan, len(by))

    raw = frame.actual_return_pct.to_numpy(float)
    resid = raw - frame.universe_mean_pct.to_numpy(float)
    out["var_share_universe"] = (float(1 - np.var(resid) / np.var(raw)), n)
    return {k: out[k] for k in POOLED_METRICS}


def statistics_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in frame.groupby("split", sort=False):
        for scope in ["pooled", "residual"]:
            for metric, (value, n) in compute_scope(g, scope).items():
                rows.append({"split": split, "scope": scope, "metric": metric,
                             "value": value, "n_obs": n})
    return pd.DataFrame(rows)


def add_vol_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """v = the stock's trailing 20-session mean of |a|, excluding the current day.
    Built over the full per-symbol history so early split rows are not penalised."""
    out = frame.sort_values(["symbol", "pred_date"]).copy()
    out["vol_baseline_20"] = out.groupby("symbol", sort=False)["actual_magnitude_pct"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    return out


def economic_robustness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Score the emitted direction on assignment and executable-timestamp targets.

    ``official_close`` is the graded target. ``last_trade_1529`` replaces the
    official 30-minute-VWAP close by the last supplied minute print. ``close_close``
    carries the same position through the following close.  The latter two are
    diagnostics, never model-selection objectives.
    """
    required = {
        "split", "pred_date", "pred_direction", "actual_return_pct",
        "session_close", "target_open", "target_close",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"economic robustness missing columns: {sorted(missing)}")
    targets = {
        "official_close": frame.actual_return_pct.astype(float),
        "last_trade_1529": (frame.target_open / frame.session_close - 1.0).astype(float) * 100.0,
        "close_close": (frame.target_close / frame.session_close - 1.0).astype(float) * 100.0,
    }
    rows = []
    for split, split_frame in frame.groupby("split", sort=False):
        direction = split_frame.pred_direction.to_numpy(float)
        dates = split_frame.pred_date.to_numpy()
        for target_name, full_target in targets.items():
            target = full_target.loc[split_frame.index].to_numpy(float)
            finite = np.isfinite(target)
            t = target[finite]
            d = direction[finite]
            used_dates = dates[finite]
            universe = pd.Series(t).groupby(used_dates).transform("mean").to_numpy()
            residual = t - universe
            for scope, actual in [("pooled", t), ("residual", residual)]:
                rows.append({
                    "split": split,
                    "target": target_name,
                    "scope": scope,
                    "direction_score": float(np.sum(d * actual) / np.sum(np.abs(actual))),
                    "directional_return_bps": float(np.mean(d * actual) * 100.0),
                    "n_obs": int(len(actual)),
                })
    return pd.DataFrame(rows)


def block_bootstrap_diff(frame: pd.DataFrame, score_a: np.ndarray, score_b: np.ndarray,
                         target: np.ndarray, block_sessions: int = 5, draws: int = 2000,
                         seed: int = 42) -> tuple[float, float, float]:
    """Paired block bootstrap on contiguous session blocks of the direction score."""
    rng = np.random.default_rng(seed)
    dates = np.sort(frame.pred_date.unique())
    blocks = [dates[i:i + block_sessions] for i in range(0, len(dates), block_sessions)]
    idx = {d: np.where(frame.pred_date.to_numpy() == d)[0] for d in dates}
    diffs = []
    for _ in range(draws):
        pick = rng.integers(0, len(blocks), len(blocks))
        rows = np.concatenate([np.concatenate([idx[d] for d in blocks[i]]) for i in pick])
        t = target[rows]
        den = np.sum(np.abs(t))
        diffs.append(np.sum(score_a[rows] * t) / den - np.sum(score_b[rows] * t) / den)
    diffs = np.asarray(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def block_bootstrap_magnitude_diff(
    frame: pd.DataFrame,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    actual_magnitude: np.ndarray,
    block_sessions: int = 5,
    draws: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired session-block bootstrap for magnitude_score(A)-magnitude_score(B)."""
    rng = np.random.default_rng(seed)
    dates = np.sort(frame.pred_date.unique())
    blocks = [dates[i:i + block_sessions] for i in range(0, len(dates), block_sessions)]
    idx = {d: np.where(frame.pred_date.to_numpy() == d)[0] for d in dates}
    diffs = []
    for _ in range(draws):
        pick = rng.integers(0, len(blocks), len(blocks))
        rows = np.concatenate([np.concatenate([idx[d] for d in blocks[i]]) for i in pick])
        actual = actual_magnitude[rows]
        denominator = np.sum(actual)
        score_a = 1.0 - np.sum(np.abs(pred_a[rows] - actual)) / denominator
        score_b = 1.0 - np.sum(np.abs(pred_b[rows] - actual)) / denominator
        diffs.append(score_a - score_b)
    values = np.asarray(diffs)
    return float(values.mean()), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def block_bootstrap_spearman_diff(
    frame: pd.DataFrame,
    score_a: np.ndarray,
    score_b: np.ndarray,
    outcome: np.ndarray,
    block_sessions: int = 5,
    draws: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired session-block bootstrap for two Spearman reliability scores."""
    rng = np.random.default_rng(seed)
    dates = np.sort(frame.pred_date.unique())
    blocks = [dates[i:i + block_sessions] for i in range(0, len(dates), block_sessions)]
    idx = {d: np.where(frame.pred_date.to_numpy() == d)[0] for d in dates}
    diffs = []
    for _ in range(draws):
        pick = rng.integers(0, len(blocks), len(blocks))
        rows = np.concatenate([np.concatenate([idx[d] for d in blocks[i]]) for i in pick])
        a = _safe_spearman(score_a[rows], outcome[rows])
        b = _safe_spearman(score_b[rows], outcome[rows])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    values = np.asarray(diffs)
    if not len(values):
        return np.nan, np.nan, np.nan
    return float(values.mean()), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
