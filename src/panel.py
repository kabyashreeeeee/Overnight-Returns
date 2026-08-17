"""Master calendar, target construction, splits, integrity checks, test quarantine.

Nothing here reads or writes anything outside `fresh/`, except read-only access to
../data and the pre-built minute feature cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd


class TestQuarantineError(RuntimeError):
    """Raised when test-period labels are requested in research mode."""


@dataclass(frozen=True)
class SplitBounds:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    train_valid_embargo: list
    valid_test_embargo: list


def load_daily(daily_dir: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(daily_dir / "*.parquet"))):
        symbol = os.path.basename(path)[:-8]
        d = pd.read_parquet(path)
        d.columns = [c.lower() for c in d.columns]
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        d = d.sort_values("date").drop_duplicates("date")
        d["symbol"] = symbol
        frames.append(d[["date", "symbol", "open", "high", "low", "close", "volume"]])
    daily = pd.concat(frames, ignore_index=True)
    daily = daily[daily.date.between(pd.Timestamp(start), pd.Timestamp(end))]
    return daily.sort_values(["symbol", "date"]).reset_index(drop=True)


def master_calendar(daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Union of all observed session dates. The exchange calendar, not a per-symbol one."""
    return pd.DatetimeIndex(sorted(daily["date"].unique()))


def build_targets(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """r(i,T) = (open(i, next master session) / close(i,T) - 1) * 100.

    The next session comes from the master calendar, never from the symbol's own
    next available row: a halted stock must not silently borrow a later session.
    """
    next_session = pd.Series(calendar[1:], index=calendar[:-1])
    panel = daily.copy()
    panel["pred_date"] = panel["date"]
    panel["target_date"] = panel["pred_date"].map(next_session)

    next_prices = daily[["symbol", "date", "open", "close"]].rename(
        columns={"date": "target_date", "open": "target_open", "close": "target_close"}
    )
    panel = panel.merge(
        next_prices, on=["symbol", "target_date"], how="left", validate="many_to_one"
    )

    ok = panel.target_date.notna() & panel.close.gt(0) & panel.target_open.gt(0)
    panel["actual_return_pct"] = np.where(ok, (panel.target_open / panel.close - 1.0) * 100.0, np.nan)
    panel["actual_magnitude_pct"] = panel.actual_return_pct.abs()
    panel["actual_direction"] = np.where(panel.actual_return_pct < 0, -1.0, 1.0)
    panel.loc[~ok, "actual_direction"] = np.nan
    return panel.drop(columns=["date"])


def assign_splits(calendar: pd.DatetimeIndex, train_end: str, valid_end: str,
                  embargo_sessions: int) -> SplitBounds:
    train_end_ts = pd.Timestamp(train_end)
    valid_end_ts = pd.Timestamp(valid_end)

    train_dates = calendar[calendar <= train_end_ts]
    last_train = train_dates[-1]
    after_train = calendar[calendar > last_train]
    if len(after_train) <= embargo_sessions:
        raise ValueError("not enough sessions after train for the embargo")
    valid_start = after_train[embargo_sessions]

    valid_dates = calendar[(calendar >= valid_start) & (calendar <= valid_end_ts)]
    last_valid = valid_dates[-1]
    after_valid = calendar[calendar > last_valid]
    if len(after_valid) <= embargo_sessions:
        raise ValueError("not enough sessions after validation for the embargo")
    test_start = after_valid[embargo_sessions]

    return SplitBounds(
        train_end=last_train,
        valid_start=valid_start,
        valid_end=last_valid,
        test_start=test_start,
        train_valid_embargo=list(after_train[:embargo_sessions]),
        valid_test_embargo=list(after_valid[:embargo_sessions]),
    )


def label_splits(panel: pd.DataFrame, bounds: SplitBounds) -> pd.DataFrame:
    out = panel.copy()
    out["split"] = pd.NA
    out.loc[out.pred_date <= bounds.train_end, "split"] = "train"
    out.loc[out.pred_date.between(bounds.valid_start, bounds.valid_end), "split"] = "valid"
    out.loc[out.pred_date >= bounds.test_start, "split"] = "test"
    return out


def enforce_quarantine(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """In research mode the test block is dropped entirely, labels and all."""
    if mode == "research":
        if (frame["split"] == "test").any():
            frame = frame[frame["split"] != "test"].copy()
        return frame
    if mode == "final-test":
        return frame
    raise TestQuarantineError(f"unknown mode {mode!r}")


def integrity_report(daily: pd.DataFrame, panel: pd.DataFrame,
                     minute: pd.DataFrame) -> pd.DataFrame:
    """Pre-modelling data checks. Returns one row per check."""
    checks = []

    def add(name, value, detail=""):
        checks.append({"check": name, "value": value, "detail": detail})

    add("symbols", daily.symbol.nunique())
    add("sessions", daily.date.nunique())
    add("daily_rows", len(daily))
    add("duplicate_symbol_date", int(daily.duplicated(["symbol", "date"]).sum()))
    add("negative_volume", int((daily.volume < 0).sum()))
    bad_bounds = (
        (daily.high < daily.low)
        | (daily.close > daily.high) | (daily.close < daily.low)
        | (daily.open > daily.high) | (daily.open < daily.low)
    )
    add("ohlc_bound_violations", int(bad_bounds.sum()))
    add("nonpositive_close", int((daily.close <= 0).sum()))

    scorable = panel.actual_return_pct.notna()
    add("scorable_rows", int(scorable.sum()))
    add("unscorable_rows", int((~scorable).sum()), "missing next-session open")

    # official close vs 15:29 minute close: a benchmark-construction fact, not an error
    j = panel.merge(minute[["pred_date", "symbol", "session_close"]],
                    on=["pred_date", "symbol"], how="inner")
    g = (j.session_close / j.close - 1.0) * 1e4
    add("close_vs_1529_mean_bp", round(float(g.mean()), 4))
    add("close_vs_1529_mean_abs_bp", round(float(g.abs().mean()), 4))
    add("close_vs_1529_p95_abs_bp", round(float(g.abs().quantile(0.95)), 4))
    return pd.DataFrame(checks)
