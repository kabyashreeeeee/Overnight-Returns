"""Expanding-window out-of-sample fold machinery.

Strictly chronological. Every OOF row is predicted by a model trained only on
sessions strictly earlier than the fold start, minus a five-session embargo.
No K-fold, no shuffling, no row-level randomisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    predict_start: pd.Timestamp
    predict_end: pd.Timestamp
    embargo: list


def build_folds(train_dates: pd.DatetimeIndex, n_folds: int, reserve_sessions: int,
                embargo_sessions: int) -> list[Fold]:
    """Reserve the final `reserve_sessions` of training for OOF prediction, split
    into `n_folds` contiguous blocks; each fold trains on everything before it."""
    dates = pd.DatetimeIndex(sorted(train_dates))
    if reserve_sessions >= len(dates):
        raise ValueError("reserve_sessions must be smaller than the training block")
    reserved = dates[-reserve_sessions:]
    blocks = np.array_split(reserved, n_folds)
    folds = []
    for i, block in enumerate(blocks):
        block = pd.DatetimeIndex(block)
        start = block[0]
        prior = dates[dates < start]
        if len(prior) <= embargo_sessions:
            raise ValueError(f"fold {i} has too little history for the embargo")
        embargo = list(prior[-embargo_sessions:])
        folds.append(Fold(index=i + 1, train_end=prior[-embargo_sessions - 1],
                          predict_start=start, predict_end=block[-1], embargo=embargo))
    return folds


def fold_masks(pred_date: pd.Series, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    train = (pred_date <= fold.train_end).to_numpy()
    predict = pred_date.between(fold.predict_start, fold.predict_end).to_numpy()
    return train, predict


def assert_no_overlap(pred_date: pd.Series, folds: list[Fold]) -> None:
    """Every OOF row must be outside its own training window, with the embargo respected."""
    for f in folds:
        tr, pr = fold_masks(pred_date, f)
        if (tr & pr).any():
            raise AssertionError(f"fold {f.index}: train and predict overlap")
        gap = pred_date[pr].min() - pred_date[tr].max()
        if pd.isna(gap):
            continue
        embargoed = pred_date[(pred_date > f.train_end) & (pred_date < f.predict_start)].nunique()
        if embargoed < len(f.embargo):
            raise AssertionError(f"fold {f.index}: embargo not respected ({embargoed} sessions)")
