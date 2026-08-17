"""Feature store.

Rule enforced throughout: any feature derived from the target r(i,T) is shifted by
at least one session before any rolling window. Same-date cross-sectional features
are permitted because they are known at the close of T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- minute-derived columns reused from the pre-built cache (all same-session, <=15:29)
MINUTE_COLS = [
    "session_open", "session_high", "session_low", "session_close",
    "rv_full_pct", "rv_morning_pct", "rv_afternoon_pct", "rv_final_60_pct", "rv_final_30_pct",
    "rsv_positive", "rsv_negative", "semivariance_imbalance",
    "return_first_hour_pct", "return_final_60_pct", "return_final_30_pct", "return_final_15_pct",
    "return_1400_close_pct", "trend_slope_pct", "path_efficiency", "path_length",
    "close_vs_vwap_pct", "vwap", "high_time_fraction", "low_time_fraction",
    "max_5m_return", "min_5m_return", "max_abs_5m_return", "positive_5m_share",
    "first_hour_volume_share", "final_60_volume_share", "final_30_volume_share",
    "final_15_volume_share", "max_5m_volume_share",
    "total_volume", "dollar_volume", "minute_coverage", "incomplete_5m_buckets",
]


def _shifted_roll(g, window, stat, minp):
    return g.transform(lambda s: getattr(s.shift(1).rolling(window, min_periods=minp), stat)())


def build_features(panel: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    df = panel.merge(minute[["pred_date", "symbol"] + MINUTE_COLS],
                     on=["pred_date", "symbol"], how="left")
    df = df.sort_values(["symbol", "pred_date"]).reset_index(drop=True)
    by = df.groupby("symbol", sort=False)

    # ---------------- 6.6 official-close benchmark component ----------------
    # g = last traded price / official close - 1. Both observable at close of T.
    # The official-close basis is an observable component of the exact
    # multiplicative return decomposition. For small returns, it is approximately
    # additive to the 15:29-to-open return. Flagged separately so it can be ablated.
    df["bench_gap_pct"] = (df.session_close / df.close - 1.0) * 100.0
    df["bench_gap_abs"] = df.bench_gap_pct.abs()

    # ---------------- 6.1 daily price and range ----------------
    df["prev_close"] = by.close.shift(1)
    df["ret_cc_pct"] = (df.close / df.prev_close - 1.0) * 100.0
    df["ret_oc_pct"] = (df.close / df.open - 1.0) * 100.0
    df["gap_into_t_pct"] = (df.open / df.prev_close - 1.0) * 100.0
    df["range_pct"] = (df.high - df.low) / df.prev_close * 100.0
    denom = (df.high - df.low).replace(0, np.nan)
    df["close_location"] = (df.close - df.low) / denom
    # Parkinson and Garman-Klass range volatility proxies
    with np.errstate(all="ignore"):
        df["parkinson"] = np.sqrt(np.log(df.high / df.low) ** 2 / (4 * np.log(2))) * 100.0
        df["garman_klass"] = np.sqrt(np.maximum(
            0.5 * np.log(df.high / df.low) ** 2
            - (2 * np.log(2) - 1) * np.log(df.close / df.open) ** 2, 0.0)) * 100.0
    for k in [2, 5, 20]:
        df[f"ret_{k}d_pct"] = (df.close / by.close.shift(k) - 1.0) * 100.0
    df["dist_20d_high_pct"] = (df.close / _shifted_roll(by.high, 20, "max", 10) - 1.0) * 100.0
    df["dist_20d_low_pct"] = (df.close / _shifted_roll(by.low, 20, "min", 10) - 1.0) * 100.0
    df["range_pct_rank_60"] = by.range_pct.transform(
        lambda s: s.shift(1).rolling(60, min_periods=20).rank(pct=True))

    # ---------------- 6.2 overnight history (target-derived -> always shifted) ----
    mag = df.groupby("symbol", sort=False)["actual_magnitude_pct"]
    ret = df.groupby("symbol", sort=False)["actual_return_pct"]
    df["mag_lag_1"] = mag.shift(1)
    df["ret_lag_1"] = ret.shift(1)
    df["ret_lag_2"] = ret.shift(2)
    for w in [5, 20, 60]:
        df[f"mag_mean_{w}"] = _shifted_roll(mag, w, "mean", max(3, w // 2))
        df[f"mag_std_{w}"] = _shifted_roll(mag, w, "std", max(3, w // 2))
        df[f"on_mean_{w}"] = _shifted_roll(ret, w, "mean", max(3, w // 2))
        df[f"on_pos_share_{w}"] = df.groupby("symbol", sort=False)["actual_return_pct"].transform(
            lambda s, w=w: s.shift(1).ge(0).rolling(w, min_periods=max(3, w // 2)).mean())
    df["mag_median_20"] = df.groupby("symbol", sort=False)["actual_magnitude_pct"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).median())
    df["mag_ewm_20"] = df.groupby("symbol", sort=False)["actual_magnitude_pct"].transform(
        lambda s: s.shift(1).ewm(halflife=20, adjust=False, min_periods=10).mean())
    df["on_sign_streak"] = df.groupby("symbol", sort=False)["actual_return_pct"].transform(
        lambda s: s.shift(1).ge(0).astype(float).groupby(
            s.shift(1).ge(0).ne(s.shift(2).ge(0)).cumsum()).cumcount() + 1)

    # ---------------- 6.3 realised volatility ----------------
    for c in ["rv_full_pct", "rv_final_30_pct", "rsv_positive", "rsv_negative"]:
        df[f"log_{c}"] = np.log1p(df[c])
    df["rv_ratio_final30"] = df.rv_final_30_pct / df.rv_full_pct.replace(0, np.nan)
    df["rv_z_60"] = (df.rv_full_pct - _shifted_roll(by.rv_full_pct, 60, "mean", 20)) / \
                    _shifted_roll(by.rv_full_pct, 60, "std", 20).replace(0, np.nan)
    df["rv_of_rv_20"] = _shifted_roll(by.rv_full_pct, 20, "std", 10)
    df["rv_mean_20"] = _shifted_roll(by.rv_full_pct, 20, "mean", 10)

    # ---------------- 6.5 volume and liquidity ----------------
    df["log_dollar_volume"] = np.log1p(df.dollar_volume)
    df["volume_ratio_20"] = df.total_volume / _shifted_roll(by.total_volume, 20, "mean", 10).replace(0, np.nan)
    df["amihud_20"] = df.groupby("symbol", sort=False).apply(
        lambda x: (x.ret_cc_pct.abs() / (x.dollar_volume / 1e7)).shift(1)
        .rolling(20, min_periods=10).mean(), include_groups=False).reset_index(level=0, drop=True)
    df["adv_20_cr"] = _shifted_roll(by.dollar_volume, 20, "mean", 10) / 1e7

    # ---------------- 6.9 calendar ----------------
    df["calendar_gap_days"] = (df.target_date - df.pred_date).dt.days
    df["weekday"] = df.pred_date.dt.weekday
    df["is_long_gap"] = (df.calendar_gap_days > 1).astype(float)
    df["month_end"] = (df.pred_date.dt.month != df.target_date.dt.month).astype(float)

    # ---------------- 6.10 history / listing ----------------
    df["history_count"] = by.cumcount()
    df["is_new_listing"] = (df.history_count < 60).astype(float)
    df["low_minute_coverage"] = (df.minute_coverage < 0.80).astype(float)

    # ---------------- 6.7 market-wide (same date, close of T) ----------------
    day = df.groupby("pred_date")
    mkt = pd.DataFrame({
        "mkt_ret_oc": day.ret_oc_pct.mean(),
        "mkt_ret_cc": day.ret_cc_pct.mean(),
        "mkt_rv": day.rv_full_pct.mean(),
        "mkt_rv_disp": day.rv_full_pct.std(),
        "mkt_ret_disp": day.ret_oc_pct.std(),
        "mkt_final60": day.return_final_60_pct.mean(),
        "mkt_breadth_up": day.ret_oc_pct.apply(lambda s: float((s > 0).mean())),
        "mkt_near_high": day.close_location.mean(),
        "mkt_coverage": day.minute_coverage.mean(),
        "mkt_bench_gap": day.bench_gap_pct.mean(),
    })
    # lagged market overnight history (target-derived -> shifted)
    mkt_on = df.groupby("pred_date").actual_return_pct.mean().sort_index()
    mkt["mkt_on_lag_1"] = mkt_on.shift(1)
    mkt["mkt_on_lag_2"] = mkt_on.shift(2)
    mkt["mkt_on_mean_20"] = mkt_on.shift(1).rolling(20, min_periods=10).mean()
    mkt["mkt_on_mean_60"] = mkt_on.shift(1).rolling(60, min_periods=20).mean()
    mkt["mkt_on_pos_share_20"] = mkt_on.shift(1).ge(0).rolling(20, min_periods=10).mean()
    mkt["mkt_on_std_20"] = mkt_on.shift(1).rolling(20, min_periods=10).std()
    df = df.merge(mkt, left_on="pred_date", right_index=True, how="left")

    # ---------------- 6.8 stock-relative (rank / z / difference) ----------------
    rel_base = ["ret_oc_pct", "rv_full_pct", "return_final_60_pct", "return_final_30_pct",
                "bench_gap_pct", "volume_ratio_20", "close_location", "range_pct",
                "semivariance_imbalance", "mag_mean_20", "close_vs_vwap_pct", "amihud_20"]
    day = df.groupby("pred_date")
    for c in rel_base:
        df[f"cs_rank_{c}"] = day[c].rank(pct=True)
        mu = day[c].transform("mean")
        sd = day[c].transform("std").replace(0, np.nan)
        df[f"cs_z_{c}"] = (df[c] - mu) / sd
    df["stock_minus_mkt_oc"] = df.ret_oc_pct - df.mkt_ret_oc
    df["stock_minus_mkt_rv"] = df.rv_full_pct - df.mkt_rv
    df["stock_minus_mkt_final60"] = df.return_final_60_pct - df.mkt_final60
    df["rel_rv_ratio"] = df.rv_full_pct / df.mkt_rv.replace(0, np.nan)

    df["n_missing_features"] = df.isna().sum(axis=1)
    return df


def feature_columns(df: pd.DataFrame, include_benchmark: bool = True) -> list[str]:
    """Everything that is a model input. Never the target or its direct transforms."""
    exclude = {
        "symbol", "pred_date", "target_date", "split",
        "open", "high", "low", "close", "volume", "prev_close", "target_open", "target_close",
        "actual_return_pct", "actual_magnitude_pct", "actual_direction",
        "session_open", "session_high", "session_low", "session_close", "vwap",
        "dollar_volume", "total_volume",
    }
    cols = [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not include_benchmark:
        cols = [c for c in cols if "bench_gap" not in c]
    return sorted(cols)


BENCHMARK_FEATURES = ["bench_gap_pct", "bench_gap_abs",
                      "cs_rank_bench_gap_pct", "cs_z_bench_gap_pct", "mkt_bench_gap"]
