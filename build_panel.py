"""Stage 1: build the canonical panel + feature store. Research mode drops test entirely."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import panel as P
from src import features as F
from src import oof as O
from src import minute as MIN


def main(config_path: Path, mode: str) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parent
    daily_dir = (root / cfg["paths"]["daily_dir"]).resolve()
    out_dir = (root / cfg["paths"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading daily ...", flush=True)
    daily = P.load_daily(daily_dir, cfg["data"]["start_date"], cfg["data"]["end_date"])
    cal = P.master_calendar(daily)
    print(f"  {len(daily)} rows, {daily.symbol.nunique()} symbols, {len(cal)} sessions")

    bounds = P.assign_splits(cal, cfg["splits"]["train_end"], cfg["splits"]["valid_end"],
                             cfg["splits"]["embargo_sessions"])
    print(f"  train<= {bounds.train_end.date()} | valid {bounds.valid_start.date()}"
          f"..{bounds.valid_end.date()} | test>= {bounds.test_start.date()}")
    print(f"  embargo1 {[d.date().isoformat() for d in bounds.train_valid_embargo]}")
    print(f"  embargo2 {[d.date().isoformat() for d in bounds.valid_test_embargo]}")

    # The full calendar identifies the exact next exchange session.  Research
    # targets and features are then built only from pre-test rows.
    model_daily = daily if mode == "final-test" else daily[daily.date < bounds.test_start].copy()
    tgt = P.label_splits(P.build_targets(model_daily, cal), bounds)
    if mode == "research":
        assert "test" not in set(tgt.split.dropna()), "test rows reached research targets"

    minute = MIN.load_or_build(
        (root / cfg["paths"]["minute_dir"]).resolve(),
        (root / cfg["paths"]["minute_cache"]).resolve(),
        sorted(model_daily.symbol.unique()),
        int(cfg["features"].get("expected_minute_bars", 375)),
        bool(cfg["features"].get("rebuild_minute_cache", False)),
    )
    minute["pred_date"] = pd.to_datetime(minute["pred_date"]).dt.normalize()
    if mode == "research":
        minute = minute[minute.pred_date < bounds.test_start].copy()

    print("integrity checks ...", flush=True)
    rep = P.integrity_report(model_daily, tgt, minute)
    rep.to_csv(out_dir / "integrity_report.csv", index=False)
    print(rep.to_string(index=False))

    print("\nbuilding features ...", flush=True)
    feat = F.build_features(tgt, minute)

    feat = P.enforce_quarantine(feat, mode)
    print(f"  mode={mode}; splits present: {sorted(feat.split.dropna().unique())}")

    # scorable rows: target exists and minimum history satisfied
    minh = cfg["features"]["min_history_sessions"]
    eligible = feat[feat.actual_return_pct.notna() & feat.split.notna()
                    & (feat.history_count >= minh)].copy()
    print(f"  feature rows {len(feat)} -> eligible {len(eligible)}")
    print(eligible.groupby("split").size().to_string())

    train_dates = pd.DatetimeIndex(sorted(eligible.loc[eligible.split == "train", "pred_date"].unique()))
    folds = O.build_folds(train_dates, cfg["oof"]["n_folds"], cfg["oof"]["reserve_sessions"],
                          cfg["oof"]["embargo_sessions"])
    O.assert_no_overlap(eligible.loc[eligible.split == "train", "pred_date"], folds)
    print("\nOOF folds (all inside train):")
    for f in folds:
        n = int(eligible.pred_date.between(f.predict_start, f.predict_end).sum())
        print(f"  fold {f.index}: train<= {f.train_end.date()}  predict "
              f"{f.predict_start.date()}..{f.predict_end.date()}  rows={n}")

    cols = F.feature_columns(eligible)
    print(f"\nfeature count: {len(cols)}")
    eligible.to_parquet(out_dir / "panel.parquet", index=False)
    pd.Series(cols).to_csv(out_dir / "feature_columns.csv", index=False, header=["feature"])
    pd.DataFrame([{"fold": f.index, "train_end": f.train_end,
                   "predict_start": f.predict_start, "predict_end": f.predict_end}
                  for f in folds]).to_csv(out_dir / "oof_folds.csv", index=False)
    print(f"\nwrote {out_dir/'panel.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--mode", choices=["research", "final-test"], default="research")
    args = ap.parse_args()
    main(args.config, args.mode)
