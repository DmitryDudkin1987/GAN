"""LightningDataModule для MNIST."""

from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 24,
        num_workers: int = 2,
        val_size: float = 0.1,
        seed: int = 42,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.seed = seed
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            full_train = datasets.MNIST(
                root=str(self.data_dir),
                train=True,
                download=False,
                transform=self.transform,
            )
            val_len = int(len(full_train) * self.val_size)
            train_len = len(full_train) - val_len
            self.train_dataset, self.val_dataset = random_split(
                full_train,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage in (None, "test", "predict"):
            self.test_dataset = datasets.MNIST(
                root=str(self.data_dir),
                train=False,
                download=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )