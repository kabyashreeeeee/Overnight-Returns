"""Deterministic 09:15-anchored minute aggregation.

The submitted pipeline must be reproducible from the supplied raw minute Parquets;
the cache is an optimisation, never an undeclared input.  A full NSE session has
375 one-minute bars, labelled 09:15 through 15:29.  Five-minute buckets use first
open, maximum high, minimum low, last close and summed volume.  Incomplete buckets
are retained and counted.
"""
from __future__ import annotations

from pathlib import Path
import glob

import numpy as np
import pandas as pd


MINUTE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parquet_symbols(directory: Path) -> list[str]:
    return sorted(Path(path).stem for path in glob.glob(str(directory / "*.parquet")))


def features_for_symbol(path: Path, expected_bars: int = 375) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=MINUTE_COLUMNS)
    raw.columns = [str(c).lower() for c in raw.columns]
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "open", "high", "low", "close"])
    raw = raw.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    minute_of_day = raw.timestamp.dt.hour * 60 + raw.timestamp.dt.minute
    raw = raw.loc[minute_of_day.between(9 * 60 + 15, 15 * 60 + 29)].copy()
    minute_of_day = raw.timestamp.dt.hour * 60 + raw.timestamp.dt.minute
    raw["date"] = raw.timestamp.dt.normalize()
    raw["bucket"] = ((minute_of_day - (9 * 60 + 15)) // 5).astype("int16")

    five = (
        raw.groupby(["date", "bucket"], sort=True, observed=True)
        .agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"),
            minute_count=("timestamp", "size"),
        )
        .reset_index()
    )
    group = five.groupby("date", sort=True, observed=True)
    five["prev_close"] = group.close.shift(1).fillna(five.open)
    valid = five.close.gt(0) & five.prev_close.gt(0)
    five["log_ret"] = np.where(valid, np.log(five.close / five.prev_close), np.nan)
    five["ret_sq"] = five.log_ret.pow(2)
    five["pos_sq"] = five.ret_sq.where(five.log_ret > 0, 0.0)
    five["neg_sq"] = five.ret_sq.where(five.log_ret < 0, 0.0)
    five["abs_path"] = (five.close - five.prev_close).abs()
    five["dollar_volume"] = five.close * five.volume.fillna(0.0)

    base = group.agg(
        session_open=("open", "first"), session_high=("high", "max"),
        session_low=("low", "min"), session_close=("close", "last"),
        minute_bar_count=("minute_count", "sum"), five_min_bar_count=("bucket", "size"),
        total_volume=("volume", "sum"), path_length=("abs_path", "sum"),
        max_5m_return=("log_ret", "max"), min_5m_return=("log_ret", "min"),
        max_abs_5m_return=("log_ret", lambda s: s.abs().max()),
        positive_5m_share=("log_ret", lambda s: float((s > 0).mean())),
        rsv_positive=("pos_sq", "sum"), rsv_negative=("neg_sq", "sum"),
        dollar_volume=("dollar_volume", "sum"),
    )
    base["rv_full_pct"] = np.sqrt(group.ret_sq.sum(min_count=1)) * 100.0

    def segment_sum(mask: pd.Series, value: str) -> pd.Series:
        return five.loc[mask].groupby("date", observed=True)[value].sum(min_count=1)

    base["rv_morning_pct"] = np.sqrt(segment_sum(five.bucket < 39, "ret_sq")) * 100.0
    base["rv_afternoon_pct"] = np.sqrt(segment_sum(five.bucket >= 39, "ret_sq")) * 100.0
    base["rv_final_60_pct"] = np.sqrt(segment_sum(five.bucket >= 63, "ret_sq")) * 100.0
    base["rv_final_30_pct"] = np.sqrt(segment_sum(five.bucket >= 69, "ret_sq")) * 100.0

    for name, mask in {
        "first_hour_volume_share": five.bucket < 12,
        "final_60_volume_share": five.bucket >= 63,
        "final_30_volume_share": five.bucket >= 69,
        "final_15_volume_share": five.bucket >= 72,
    }.items():
        base[name] = segment_sum(mask, "volume") / base.total_volume.replace(0, np.nan)
    base["max_5m_volume_share"] = group.volume.max() / base.total_volume.replace(0, np.nan)

    close_by_bucket = five.set_index(["date", "bucket"]).close
    for name, prior_bucket in {
        "return_final_15_pct": 71,
        "return_final_30_pct": 68,
        "return_final_60_pct": 62,
        "return_1400_close_pct": 56,
    }.items():
        prior = close_by_bucket.xs(prior_bucket, level="bucket", drop_level=True)
        base[name] = (base.session_close / prior - 1.0) * 100.0
    first_hour_close = close_by_bucket.xs(11, level="bucket", drop_level=True)
    base["return_first_hour_pct"] = (first_hour_close / base.session_open - 1.0) * 100.0

    high_idx = group.high.idxmax()
    low_idx = group.low.idxmin()
    base["high_time_fraction"] = five.loc[high_idx].set_index("date").bucket / 74.0
    base["low_time_fraction"] = five.loc[low_idx].set_index("date").bucket / 74.0
    base["vwap"] = base.dollar_volume / base.total_volume.replace(0, np.nan)
    base["close_vs_vwap_pct"] = (base.session_close / base.vwap - 1.0) * 100.0
    base["path_efficiency"] = (
        (base.session_close - base.session_open).abs() / base.path_length.replace(0, np.nan)
    )

    five["x"] = five.bucket / 74.0
    five["log_close"] = np.log(five.close.where(five.close > 0))
    five["x_centered"] = five.x - group.x.transform("mean")
    five["y_centered"] = five.log_close - group.log_close.transform("mean")
    numerator = (five.x_centered * five.y_centered).groupby(five.date).sum()
    denominator = five.x_centered.pow(2).groupby(five.date).sum()
    base["trend_slope_pct"] = numerator / denominator.replace(0, np.nan) * 100.0

    base["semivariance_imbalance"] = (
        (base.rsv_positive - base.rsv_negative)
        / (base.rsv_positive + base.rsv_negative).replace(0, np.nan)
    )
    base["rsv_positive"] = np.sqrt(base.rsv_positive) * 100.0
    base["rsv_negative"] = np.sqrt(base.rsv_negative) * 100.0
    for col in ["max_5m_return", "min_5m_return", "max_abs_5m_return"]:
        base[col] *= 100.0
    base["minute_coverage"] = base.minute_bar_count / float(expected_bars)
    base["incomplete_5m_buckets"] = group.minute_count.apply(lambda s: int((s < 5).sum()))
    base["symbol"] = path.stem
    base.index.name = "pred_date"
    return base.reset_index()


def load_or_build(
    minute_dir: Path,
    cache_path: Path,
    expected_symbols: list[str],
    expected_bars: int = 375,
    rebuild: bool = False,
) -> pd.DataFrame:
    minute_symbols = parquet_symbols(minute_dir)
    missing = sorted(set(expected_symbols) - set(minute_symbols))
    if missing:
        raise FileNotFoundError(
            f"minute data missing for {len(missing)} daily symbols: {', '.join(missing[:10])}"
        )
    if cache_path.exists() and not rebuild:
        cached = pd.read_parquet(cache_path)
        cached["pred_date"] = pd.to_datetime(cached.pred_date).dt.normalize()
        if set(expected_symbols).issubset(set(cached.symbol.unique())):
            return cached

    frames = []
    for number, symbol in enumerate(expected_symbols, 1):
        frames.append(features_for_symbol(minute_dir / f"{symbol}.parquet", expected_bars))
        if number == 1 or number % 10 == 0 or number == len(expected_symbols):
            print(f"aggregated minute features for {number}/{len(expected_symbols)} symbols", flush=True)
    result = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    return result
