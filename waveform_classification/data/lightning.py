# pyright: reportUnknownVariableType=false, reportAttributeAccessIssue=false
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray

from waveform_classification.data.utils import stratified_split

type WaveformDataitem = tuple[torch.Tensor, torch.Tensor]


def _use_pin_memory() -> bool:
    return torch.cuda.is_available()


class WaveformDataset(Dataset[WaveformDataitem]):
    def __init__(self, waveforms: torch.Tensor, labels: torch.Tensor):
        self.waveforms = waveforms
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> WaveformDataitem:
        return self.waveforms[idx], self.labels[idx]


class Data(pl.LightningDataModule):
    name = "lightning"
    input_dim = 1000

    classnames = (
        "-CG",
        "CG-IR",
        "NBE",
        "+CG",
        "CC",
        "+PBP",
        "-PBP",
        "-NBE",
        "+NBE",
        "Sky Wave",
    )

    def __init__(
        self,
        data_dir: str = "datasets/waveform/lightning",
        batch_size: int = 32,
        seed: int = 42,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seed = seed
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.n_class = len(self.classnames)

        waveforms: list[np.ndarray] = []
        labels = []
        files = [
            "01.npy",
            "02.npy",
            "03.npy",
            "04.npy",
            "05.npy",
            "06.npy",
            "07.npy",
            "08.npy",
            "09.npy",
            "010.npy",
        ]

        for label, fname in enumerate(files):
            arr = np.load(self.data_dir / fname)
            waveforms.append(arr)
            labels.extend([label] * len(arr))

        self.waveforms = np.concatenate(waveforms, axis=0).astype(np.float32)
        self.labels = np.array(labels, dtype=np.int64)

    @property
    def class_weights(self) -> torch.Tensor:
        if not hasattr(self, "_class_weights"):
            msg = "class_weights not available until setup() is called"
            raise RuntimeError(msg)
        return self._class_weights

    def setup(self, stage: str | None = None):
        train_idx, val_idx, test_idx = stratified_split(
            self.labels,
            self.train_frac,
            self.val_frac,
            self.seed,
        )

        # Compute class weights from TRAINING data only (no data leakage)
        weights = cast(
            "NDArray[np.float64]",
            compute_class_weight("balanced", classes=np.arange(self.n_class), y=self.labels[train_idx]),
        )
        self._class_weights = torch.from_numpy(weights.astype(np.float32))

        self.scaler = StandardScaler()
        self.scaler.fit(self.waveforms[train_idx])
        normalized = self.scaler.transform(self.waveforms).astype(np.float32)

        self._train = WaveformDataset(
            torch.from_numpy(normalized[train_idx]),
            torch.from_numpy(self.labels[train_idx]),
        )
        self._val = WaveformDataset(
            torch.from_numpy(normalized[val_idx]),
            torch.from_numpy(self.labels[val_idx]),
        )
        self._test = WaveformDataset(
            torch.from_numpy(normalized[test_idx]),
            torch.from_numpy(self.labels[test_idx]),
        )

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=_use_pin_memory(),
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=_use_pin_memory(),
        )

    def test_dataloader(self):
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=_use_pin_memory(),
        )

    def predict_dataloader(self):
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=_use_pin_memory(),
        )
