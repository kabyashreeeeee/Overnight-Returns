"""Independent verification of the mechanically graded submission interface."""
from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import emit as E
from . import metrics as MET


def verify_submission(out_dir: Path) -> dict[str, object]:
    problems = E.validate_schema(out_dir)
    if problems:
        raise AssertionError(f"schema verification failed: {problems}")

    predictions = pd.read_csv(out_dir / "predictions.csv", parse_dates=["pred_date", "target_date"])
    actuals = pd.read_csv(out_dir / "actuals.csv", parse_dates=["pred_date", "target_date"])
    statistics = pd.read_csv(out_dir / "statistics.csv")
    keys = ["pred_date", "target_date", "symbol"]
    if not predictions[keys].equals(actuals[keys]):
        raise AssertionError("prediction and actual key panels differ")
    if not np.allclose(actuals.actual_magnitude_pct, actuals.actual_return_pct.abs(), atol=5e-10):
        raise AssertionError("actual_magnitude_pct is not abs(actual_return_pct)")
    expected_direction = np.where(actuals.actual_return_pct < 0, -1, 1)
    if not np.array_equal(actuals.actual_direction.to_numpy(), expected_direction):
        raise AssertionError("actual_direction is inconsistent with actual_return_pct")
    expected_mean = actuals.groupby("pred_date").actual_return_pct.transform("mean")
    if not np.allclose(actuals.universe_mean_pct, expected_mean, atol=5e-10):
        raise AssertionError("universe_mean_pct is inconsistent with the submitted rows")

    scored = predictions.merge(actuals, on=keys, validate="1:1")
    recomputed = MET.statistics_table(MET.add_vol_baseline(scored)).reset_index(drop=True)
    left = statistics.sort_values(["split", "scope", "metric"]).reset_index(drop=True)
    right = recomputed.sort_values(["split", "scope", "metric"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)

    patterns = [re.compile(r"\d+\.\d{4}"), re.compile(r"\d+\.\d{6}"), re.compile(r"\d+\.\d{6}")]
    bad = [0, 0, 0]
    with open(out_dir / "predictions.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for index, column in enumerate(["pred_magnitude_pct", "conf_direction", "conf_magnitude"]):
                bad[index] += not bool(patterns[index].fullmatch(row[column]))
    if any(bad):
        raise AssertionError(f"serialized prediction precision mismatch: {bad}")

    return {
        "rows": len(predictions),
        "statistics_rows": len(statistics),
        "splits": predictions.groupby("split").size().to_dict(),
    }


def verify_zip(zip_path: Path, package_name: str) -> dict[str, object]:
    required = {"research.pdf", "predictions.csv", "actuals.csv", "statistics.csv", "code"}
    cruft_tokens = ("__pycache__", ".pyc", ".DS_Store", ".pytest_cache",
                    f"{package_name}/code/outputs", "/.")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    prefix = f"{package_name}/"
    if not all(name.startswith(prefix) for name in names):
        raise AssertionError("zip contains entries outside the required package directory")
    roots = {
        name[len(prefix):].split("/", 1)[0]
        for name in names if name != prefix and name[len(prefix):]
    }
    if roots != required:
        raise AssertionError(f"zip root entries {sorted(roots)} != {sorted(required)}")
    cruft = [name for name in names if any(token in name for token in cruft_tokens)]
    if cruft:
        raise AssertionError(f"zip contains cruft: {cruft[:10]}")
    return {
        "root_entries": sorted(roots),
        "python_files": sum(name.endswith(".py") for name in names),
        "archive_entries": len(names),
    }
