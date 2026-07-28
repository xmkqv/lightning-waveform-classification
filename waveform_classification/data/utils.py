from typing import NamedTuple, NewType, cast

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

TrainIndices = NewType("TrainIndices", np.ndarray)
ValIndices = NewType("ValIndices", np.ndarray)
TestIndices = NewType("TestIndices", np.ndarray)


class SplitIndices(NamedTuple):
    train: TrainIndices
    val: ValIndices
    test: TestIndices


def stratified_split(
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> SplitIndices:
    """Split indices preserving class distribution in each split.

    Uses sklearn's train_test_split with stratify for robust handling
    of imbalanced classes.
    """
    n = len(labels)
    indices = np.arange(n)

    # First split: train vs (val + test)
    train_idx, temp_idx = cast(
        "tuple[np.ndarray, np.ndarray]",
        train_test_split(
            indices,
            train_size=train_frac,
            random_state=seed,
            stratify=labels,
        ),
    )

    # Second split: val vs test from remaining
    remaining_frac = 1.0 - train_frac
    val_ratio = val_frac / remaining_frac

    val_idx, test_idx = cast(
        "tuple[np.ndarray, np.ndarray]",
        train_test_split(
            temp_idx,
            train_size=val_ratio,
            random_state=seed,
            stratify=labels[temp_idx],
        ),
    )

    return SplitIndices(
        TrainIndices(train_idx),
        ValIndices(val_idx),
        TestIndices(test_idx),
    )
