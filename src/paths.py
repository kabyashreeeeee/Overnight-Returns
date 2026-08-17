"""Project paths that work both in development and inside the submitted ZIP."""
from __future__ import annotations

from pathlib import Path


PACKAGE_NAME = "Kabyashree_Dey_QuantIntern_EquityDesk"


def submission_dir(project_root: Path) -> Path:
    project_root = project_root.resolve()
    if project_root.name == "code" and project_root.parent.name == PACKAGE_NAME:
        return project_root.parent
    return project_root / "submission" / PACKAGE_NAME
