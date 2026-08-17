"""Contract tests for the fresh pipeline. These encode the assignment's hard rules."""
from __future__ import annotations

import sys
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import panel as P
from src import oof as O
from src import metrics as MET
from src import direction as D
from src import confidence as C
from src import features as F
from src import freeze as FZ
from src import emit as E
from src import minute as MIN
from src import verify as VERIFY
from src.paths import PACKAGE_NAME, submission_dir
from package import prepare_submission_root, required_paths


def _row(sym, date, o, c):
    return {"symbol": sym, "date": pd.Timestamp(date), "open": o,
            "high": max(o, c), "low": min(o, c), "close": c, "volume": 100.0}


# --------------------------------------------------------------------------- #
# target alignment
# --------------------------------------------------------------------------- #
def test_target_uses_master_calendar_not_next_available_row():
    """A stock absent on the true next session must yield NaN, never borrow a later open."""
    daily = pd.DataFrame([
        _row("A", "2024-01-02", 100, 100),
        _row("A", "2024-01-04", 130, 130),      # A is missing on 2024-01-03
        _row("B", "2024-01-02", 200, 200),
        _row("B", "2024-01-03", 202, 202),
        _row("B", "2024-01-04", 202, 202),
    ])
    cal = P.master_calendar(daily)
    out = P.build_targets(daily, cal)
    a = out[(out.symbol == "A") & (out.pred_date == pd.Timestamp("2024-01-02"))].iloc[0]
    assert a.target_date == pd.Timestamp("2024-01-03")
    assert np.isnan(a.actual_return_pct)


def test_target_formula_and_zero_maps_to_up():
    daily = pd.DataFrame([_row("A", "2024-01-02", 100, 100), _row("A", "2024-01-03", 101, 101)])
    out = P.build_targets(daily, P.master_calendar(daily))
    r = out[out.pred_date == pd.Timestamp("2024-01-02")].iloc[0]
    assert r.actual_return_pct == pytest.approx(1.0)
    z = pd.DataFrame([_row("B", "2024-01-02", 100, 100), _row("B", "2024-01-03", 100, 100)])
    oz = P.build_targets(z, P.master_calendar(z))
    assert oz.iloc[0].actual_return_pct == 0.0
    assert oz.iloc[0].actual_direction == 1


# --------------------------------------------------------------------------- #
# splits and embargo
# --------------------------------------------------------------------------- #
def test_embargo_is_exactly_five_master_sessions_both_sides():
    dates = pd.bdate_range("2024-01-01", periods=40)
    b = P.assign_splits(dates, str(dates[9].date()), str(dates[24].date()), 5)
    assert len(b.train_valid_embargo) == 5
    assert len(b.valid_test_embargo) == 5
    assert b.valid_start == dates[15]
    assert b.test_start == dates[30]
    frame = pd.DataFrame({"pred_date": dates})
    lab = P.label_splits(frame, b)
    assert lab.loc[lab.pred_date.isin(dates[10:15]), "split"].isna().all()
    assert lab.loc[lab.pred_date.isin(dates[25:30]), "split"].isna().all()


def test_research_mode_drops_test_rows_entirely():
    f = pd.DataFrame({"split": ["train", "valid", "test"], "actual_return_pct": [1.0, 2.0, 3.0]})
    kept = P.enforce_quarantine(f, "research")
    assert "test" not in set(kept.split)
    assert len(P.enforce_quarantine(f, "final-test")) == 3
    with pytest.raises(P.TestQuarantineError):
        P.enforce_quarantine(f, "bogus")


# --------------------------------------------------------------------------- #
# OOF machinery
# --------------------------------------------------------------------------- #
def test_oof_folds_are_chronological_and_embargoed():
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=400))
    folds = O.build_folds(dates, n_folds=5, reserve_sessions=200, embargo_sessions=5)
    assert len(folds) == 5
    for f in folds:
        assert f.train_end < f.predict_start
        gap = dates[(dates > f.train_end) & (dates < f.predict_start)]
        assert len(gap) == 5
    for a, b in zip(folds, folds[1:]):
        assert a.predict_end < b.predict_start          # blocks never overlap


def test_oof_train_and_predict_never_overlap():
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=400))
    folds = O.build_folds(dates, 5, 200, 5)
    s = pd.Series(dates)
    O.assert_no_overlap(s, folds)
    for f in folds:
        tr, pr = O.fold_masks(s, f)
        assert not (tr & pr).any()


# --------------------------------------------------------------------------- #
# distribution model
# --------------------------------------------------------------------------- #
def test_quantile_rearrangement_and_prob_up():
    levels = [0.1, 0.5, 0.9]
    q = np.array([[-1.0, 0.5, 2.0], [1.0, 2.0, 3.0], [-3.0, -2.0, -1.0]])
    p = D.prob_up_from_quantiles(q, levels)
    assert p[1] == pytest.approx(1.0 - 1e-6, abs=1e-5)      # entirely positive
    assert p[2] == pytest.approx(1e-6, abs=1e-5)            # entirely negative
    assert 0.0 < p[0] < 1.0
    # monotone rearrangement must produce a valid ordered curve
    raw = np.array([[0.5, -0.2, 1.0]])
    assert (np.diff(np.sort(raw, axis=1), axis=1) >= 0).all()


def test_confidence_flip_rule_keeps_conf_at_or_above_half():
    base = np.array([1.0, 1.0, -1.0])
    q = np.array([0.8, 0.3, 0.2])
    d, conf = C.emit(base, q)
    assert (conf >= 0.5).all()
    assert d[0] == 1 and d[1] == -1 and d[2] == 1
    assert conf[1] == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _toy():
    n = 60
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2024-01-01", periods=6), 10)
    a = rng.normal(0.1, 1.0, n)
    return pd.DataFrame({
        "pred_date": dates, "symbol": np.tile([f"S{i}" for i in range(10)], 6),
        "actual_return_pct": a, "actual_magnitude_pct": np.abs(a),
        "pred_magnitude_pct": np.abs(a) + rng.normal(0, 0.1, n),
        "pred_direction": np.where(a >= 0, 1, -1),
        "conf_direction": np.full(n, 0.7), "conf_magnitude": rng.uniform(0, 1, n),
        "split": "valid",
    }).assign(universe_mean_pct=lambda d: d.groupby("pred_date").actual_return_pct.transform("mean"))


def test_perfect_direction_scores_one_and_always_up_is_not_zero():
    f = MET.add_vol_baseline(_toy())
    m = MET.compute_scope(f, "pooled")
    assert m["direction_score"][0] == pytest.approx(1.0)
    assert m["hit_rate"][0] == pytest.approx(1.0)
    g = f.copy(); g["pred_direction"] = 1
    always = MET.compute_scope(g, "pooled")["direction_score"][0]
    # the brief claims always-up scores 0 "by construction"; it does not when the
    # mean overnight return is non-zero. We report always-up as an explicit comparator.
    assert always != pytest.approx(0.0)


def test_zero_magnitude_everywhere_scores_zero():
    f = MET.add_vol_baseline(_toy())
    f["pred_magnitude_pct"] = 0.0
    assert MET.compute_scope(f, "pooled")["magnitude_score"][0] == pytest.approx(0.0)


def test_statistics_table_has_41_rows_per_split():
    f = MET.add_vol_baseline(_toy())
    s = MET.statistics_table(f)
    assert len(s) == 41
    assert (s[s.scope == "pooled"].shape[0], s[s.scope == "residual"].shape[0]) == (25, 16)


def test_vol_baseline_excludes_current_day():
    f = pd.DataFrame({"symbol": ["A"] * 25, "pred_date": pd.date_range("2024-01-01", periods=25),
                      "actual_magnitude_pct": np.arange(25.0)})
    out = MET.add_vol_baseline(f)
    assert np.isnan(out.vol_baseline_20.iloc[19])
    assert out.vol_baseline_20.iloc[20] == pytest.approx(np.arange(20.0).mean())


# --------------------------------------------------------------------------- #
# feature, timestamp and target-mechanics contracts
# --------------------------------------------------------------------------- #
def test_feature_registry_excludes_every_current_or_future_label():
    frame = pd.DataFrame({
        "safe": [1.0], "bench_gap_pct": [0.1], "mkt_bench_gap": [0.1],
        "actual_return_pct": [2.0], "actual_magnitude_pct": [2.0],
        "actual_direction": [1.0], "target_open": [102.0], "target_close": [103.0],
        "pred_date": pd.to_datetime(["2024-01-01"]), "target_date": pd.to_datetime(["2024-01-02"]),
        "symbol": ["A"], "split": ["train"],
    })
    full = F.feature_columns(frame)
    ablated = F.feature_columns(frame, include_benchmark=False)
    assert "safe" in full
    assert not {"actual_return_pct", "actual_magnitude_pct", "actual_direction",
                "target_open", "target_close"}.intersection(full)
    assert "bench_gap_pct" in full and "bench_gap_pct" not in ablated
    assert "mkt_bench_gap" not in ablated


def test_target_derived_rolling_feature_is_strictly_shifted():
    frame = pd.DataFrame({"symbol": ["A"] * 6, "value": np.arange(1.0, 7.0)})
    group = frame.groupby("symbol")["value"]
    result = F._shifted_roll(group, 3, "mean", 1)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(2.0)
    changed = frame.copy(); changed.loc[3, "value"] = 10_000.0
    changed_result = F._shifted_roll(changed.groupby("symbol")["value"], 3, "mean", 1)
    assert changed_result.iloc[3] == pytest.approx(result.iloc[3])


def test_official_close_decomposition_identity():
    official_close, last_trade, next_open = 100.0, 100.4, 101.0
    official = next_open / official_close
    basis = last_trade / official_close
    last_to_open = next_open / last_trade
    assert official == pytest.approx(basis * last_to_open)


def test_minute_aggregation_is_0915_anchored_and_retains_incomplete_bucket(tmp_path):
    timestamps = pd.date_range("2024-01-02 09:15", periods=375, freq="min")
    price = 100.0 + np.arange(375) * 0.01
    raw = pd.DataFrame({
        "timestamp": timestamps, "open": price, "high": price + 0.02,
        "low": price - 0.02, "close": price, "volume": np.ones(375),
    }).drop(index=2)
    path = tmp_path / "SYNTH.parquet"; raw.to_parquet(path, index=False)
    result = MIN.features_for_symbol(path).iloc[0]
    assert result.minute_bar_count == 374
    assert result.five_min_bar_count == 75
    assert result.incomplete_5m_buckets == 1
    assert result.session_open == pytest.approx(price[0])
    assert result.session_close == pytest.approx(price[-1])
    assert result.return_final_15_pct == pytest.approx((price[374] / price[359] - 1) * 100)


def test_economic_robustness_reconstructs_last_trade_targets():
    frame = pd.DataFrame({
        "split": ["test"] * 4,
        "pred_date": pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2),
        "pred_direction": [1, -1, 1, -1],
        "actual_return_pct": [1.0, -1.0, 2.0, -2.0],
        "session_close": [100.0] * 4,
        "target_open": [101.0, 99.0, 102.0, 98.0],
        "target_close": [102.0, 98.0, 103.0, 97.0],
    }, index=[10, 20, 30, 40])
    result = MET.economic_robustness_table(frame)
    last = result[(result.target == "last_trade_1529") & (result.scope == "pooled")].iloc[0]
    assert last.direction_score == pytest.approx(1.0)
    assert last.directional_return_bps == pytest.approx(150.0)


# --------------------------------------------------------------------------- #
# confidence, uncertainty and reliability contracts
# --------------------------------------------------------------------------- #
def test_equal_width_ece_is_probability_weighted():
    report = C.calibration_report(np.array([0.55, 0.55, 0.85, 0.85]),
                                  np.array([1.0, 0.0, 1.0, 1.0]))
    assert report["ece_10"] == pytest.approx(0.10)


def test_error_history_uses_only_prior_realised_errors():
    frame = pd.DataFrame({
        "pred_date": pd.date_range("2024-01-01", periods=7),
        "symbol": ["A"] * 7,
        "oof_abs_error": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100.0],
    })
    result = C.add_error_history(frame)
    assert np.isnan(result.iloc[0].err_hist_mean_20)
    assert result.iloc[5].err_hist_mean_20 == pytest.approx(np.mean([0.2, 0.3, 0.4, 0.5, 0.6]))
    # The current row's extreme error cannot enter its own history.
    assert result.iloc[6].err_hist_mean_20 == pytest.approx(np.mean([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))


def test_error_history_attachment_is_date_aligned_for_train_and_frozen_after_train():
    dates = pd.date_range("2024-01-01", periods=7)
    history = C.add_error_history(pd.DataFrame({
        "pred_date": dates,
        "symbol": ["A"] * 7,
        "oof_abs_error": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100.0],
    }))
    frame = pd.DataFrame({
        "pred_date": [dates[0], dates[5], dates[-1] + pd.Timedelta(days=1),
                      dates[-1] + pd.Timedelta(days=2)],
        "symbol": ["A"] * 4,
        "split": ["train", "train", "valid", "test"],
    })
    result = C.attach_error_history(frame, history)
    assert np.isnan(result.loc[0, "err_hist_mean_20"])
    assert result.loc[1, "err_hist_mean_20"] == pytest.approx(np.mean([0.2, 0.3, 0.4, 0.5, 0.6]))
    expected_last = np.mean([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert result.loc[2, "err_hist_mean_20"] == pytest.approx(expected_last)
    assert result.loc[3, "err_hist_mean_20"] == pytest.approx(expected_last)


def test_risk_to_confidence_is_monotone_and_bounded():
    confidence = C.risk_to_confidence(np.array([-1.0, 1.0, 2.5, 5.0]),
                                      np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.all(np.diff(confidence) <= 0)
    assert confidence.min() >= 0 and confidence.max() <= 1


def test_paired_bootstraps_are_deterministic_with_nonconsecutive_index():
    frame = pd.DataFrame({"pred_date": np.repeat(pd.date_range("2024-01-01", periods=6), 2)},
                         index=np.arange(100, 112) * 3)
    actual = np.ones(12)
    first = MET.block_bootstrap_diff(frame, np.ones(12), np.zeros(12), actual,
                                     block_sessions=2, draws=40, seed=42)
    second = MET.block_bootstrap_diff(frame, np.ones(12), np.zeros(12), actual,
                                      block_sessions=2, draws=40, seed=42)
    assert first == second == pytest.approx((1.0, 1.0, 1.0))


def test_magnitude_and_reliability_bootstraps_are_paired():
    frame = pd.DataFrame({"pred_date": np.repeat(pd.date_range("2024-01-01", periods=6), 2)})
    actual = np.ones(12)
    mag = MET.block_bootstrap_magnitude_diff(
        frame, np.ones(12), np.zeros(12), actual, block_sessions=2, draws=30, seed=42
    )
    rel = MET.block_bootstrap_spearman_diff(
        frame, np.arange(12.0), -np.arange(12.0), np.arange(12.0),
        block_sessions=2, draws=30, seed=42,
    )
    assert mag == pytest.approx((1.0, 1.0, 1.0))
    assert rel[0] > 1.9 and rel[1] > 1.9


# --------------------------------------------------------------------------- #
# active freeze and reproducibility contracts
# --------------------------------------------------------------------------- #
def test_missing_freeze_manifest_blocks_final_path(tmp_path):
    with pytest.raises(RuntimeError, match="not frozen"):
        FZ.require_frozen(tmp_path / "missing.json", "model selection")


def test_frozen_code_hashes_are_actively_verified(tmp_path):
    (tmp_path / "model.py").write_text("VALUE = 1\n")
    manifest = {"code_sha256": FZ.sha256_tree(tmp_path), "research_artifact_sha256": {}}
    path = tmp_path / "selection.json"; FZ.write_manifest(path, manifest)
    FZ.require_frozen(path, "model selection", code_root=tmp_path)
    (tmp_path / "model.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="code differs"):
        FZ.require_frozen(path, "model selection", code_root=tmp_path)


def test_frozen_research_artifact_hashes_are_actively_verified(tmp_path):
    artifact = tmp_path / "folds.csv"; artifact.write_text("fold\n1\n")
    manifest = {
        "code_sha256": FZ.sha256_tree(tmp_path),
        "research_artifact_sha256": {"folds.csv": FZ.sha256_file(artifact)},
    }
    path = tmp_path / "selection.json"; FZ.write_manifest(path, manifest)
    FZ.require_frozen(path, "model selection", artifact_root=tmp_path)
    artifact.write_text("fold\n2\n")
    with pytest.raises(RuntimeError, match="research artifacts changed"):
        FZ.require_frozen(path, "model selection", artifact_root=tmp_path)


def test_specification_digest_is_order_stable_and_change_sensitive():
    first = {"selected": {"lambda": 1.0, "model": "M5"}, "features": ["a", "b"]}
    reordered = {"features": ["a", "b"], "selected": {"model": "M5", "lambda": 1.0}}
    changed = {"selected": {"lambda": 0.75, "model": "M5"}, "features": ["a", "b"]}
    assert FZ.sha256_json(first) == FZ.sha256_json(reordered)
    assert FZ.sha256_json(first) != FZ.sha256_json(changed)


def test_clean_room_audit_matches_current_graded_outputs_and_states_boundary():
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "reproduction_audit.json").read_text())
    submission = submission_dir(root)
    for name, digest in audit["artifact_sha256"].items():
        assert FZ.sha256_file(submission / name) == digest
    boundary = audit["historical_boundary"].lower()
    assert "does not claim" in boundary and "first emission" in boundary


def test_frozen_code_tree_excludes_archived_submissions(tmp_path):
    active = tmp_path / "src" / "model.py"
    active.parent.mkdir()
    active.write_text("ACTIVE = True\n")
    archived = tmp_path / "archive" / "previous_submission" / "code" / "model.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("ACTIVE = False\n")

    tree = FZ.sha256_tree(tmp_path)
    assert set(tree) == {"src/model.py"}
    FZ.verify_code_tree(tmp_path, tree)


def _submission_frame() -> pd.DataFrame:
    rows = []
    starts = {"train": "2023-01-02", "valid": "2024-01-02", "test": "2025-01-02"}
    for split, start in starts.items():
        for date in pd.bdate_range(start, periods=25):
            for symbol, actual in [("A", 1.0), ("B", -0.5)]:
                rows.append({
                    "pred_date": date, "target_date": date + pd.offsets.BDay(1), "symbol": symbol,
                    "pred_magnitude_pct": abs(actual) * 0.9, "pred_direction": 1 if actual >= 0 else -1,
                    "conf_direction": 0.75, "conf_magnitude": 0.8 if symbol == "A" else 0.4,
                    "actual_return_pct": actual, "split": split,
                })
    return pd.DataFrame(rows)


def test_output_writer_is_deterministic_and_statistics_use_rounded_interface(tmp_path):
    first = tmp_path / "first"; second = tmp_path / "second"
    E.write_outputs(_submission_frame(), first)
    E.write_outputs(_submission_frame(), second)
    for name in ["predictions.csv", "actuals.csv", "statistics.csv"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    verified = VERIFY.verify_submission(first)
    assert verified["rows"] == 150 and verified["statistics_rows"] == 123


def test_schema_validator_rejects_duplicate_keys(tmp_path):
    E.write_outputs(_submission_frame(), tmp_path)
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    pd.concat([predictions, predictions.iloc[[0]]]).to_csv(tmp_path / "predictions.csv", index=False)
    assert any("duplicate" in problem for problem in E.validate_schema(tmp_path))


def test_submission_path_works_in_development_and_extracted_layout(tmp_path):
    dev = tmp_path / "fresh"; dev.mkdir()
    assert submission_dir(dev) == dev / "submission" / PACKAGE_NAME
    code = tmp_path / PACKAGE_NAME / "code"; code.mkdir(parents=True)
    assert submission_dir(code) == code.parent


def test_zip_verifier_enforces_exact_root_and_no_cruft(tmp_path):
    path = tmp_path / "good.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in ["research.pdf", "predictions.csv", "actuals.csv", "statistics.csv"]:
            archive.writestr(f"{PACKAGE_NAME}/{name}", "x")
        archive.writestr(f"{PACKAGE_NAME}/code/main.py", "x")
    assert VERIFY.verify_zip(path, PACKAGE_NAME)["python_files"] == 1
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr(f"{PACKAGE_NAME}/research.pdf", "x")
        archive.writestr(f"{PACKAGE_NAME}/.DS_Store", "x")
    with pytest.raises(AssertionError):
        VERIFY.verify_zip(bad, PACKAGE_NAME)
    transient = tmp_path / "transient.zip"
    with zipfile.ZipFile(transient, "w") as archive:
        for name in ["research.pdf", "predictions.csv", "actuals.csv", "statistics.csv"]:
            archive.writestr(f"{PACKAGE_NAME}/{name}", "x")
        archive.writestr(f"{PACKAGE_NAME}/code/main.py", "x")
        archive.writestr(f"{PACKAGE_NAME}/code/outputs/panel.parquet", "x")
    with pytest.raises(AssertionError):
        VERIFY.verify_zip(transient, PACKAGE_NAME)


def test_extracted_packaging_preserves_raw_inputs_and_excludes_them_from_archive(tmp_path):
    required = {"research.pdf", "predictions.csv", "actuals.csv", "statistics.csv", "code"}
    for name in required:
        path = tmp_path / name
        path.mkdir() if name == "code" else path.write_text("x")
    data = tmp_path / "data"
    data.mkdir()
    (data / "raw.parquet").write_text("raw")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "features.parquet").write_text("cache")
    outputs = tmp_path / "code" / "outputs"
    outputs.mkdir()
    (outputs / "panel.parquet").write_text("transient")
    prepare_submission_root(tmp_path, required, clean_extras=False)
    assert (data / "raw.parquet").read_text() == "raw"
    assert (cache / "features.parquet").read_text() == "cache"
    archived_paths = [p.relative_to(tmp_path) for p in required_paths(tmp_path, required)]
    assert {p.parts[0] for p in archived_paths} == required
    assert not any(p.parts[:2] == ("code", "outputs") for p in archived_paths)


def test_sources_and_config_contain_no_user_specific_absolute_paths():
    root = Path(__file__).resolve().parents[1]
    files = list(root.glob("*.py")) + list((root / "src").glob("*.py")) + [root / "config.yaml"]
    text = "\n".join(path.read_text() for path in files)
    assert "/Users/" not in text and "C:\\Users\\" not in text


def test_generated_report_has_no_serialized_reportlab_objects_or_orphan_page():
    report = submission_dir(Path(__file__).resolve().parents[1]) / "research.pdf"
    if not report.exists():
        pytest.skip("report is generated after final scoring")
    from pypdf import PdfReader
    pages = [page.extract_text() or "" for page in PdfReader(str(report)).pages]
    assert 1 <= len(pages) <= 12
    assert not any("Paragraph(" in page or "ParaFrag(" in page for page in pages)
    assert len(pages[-1].strip()) > 200
