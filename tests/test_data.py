"""Tests for waveform loading and splitting."""

import numpy as np

from waveform_classification.data.utils import stratified_split


def test_stratified_split_is_disjoint_and_preserves_classes() -> None:
    labels = np.repeat(np.arange(10), 20)

    split = stratified_split(labels, train_frac=0.6, val_frac=0.2, seed=7)

    assert len(split.train) == 120
    assert len(split.val) == 40
    assert len(split.test) == 40
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)
    assert set(np.unique(labels[split.train])) == set(range(10))
    assert set(np.unique(labels[split.val])) == set(range(10))
    assert set(np.unique(labels[split.test])) == set(range(10))
