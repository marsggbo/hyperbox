from typing import Optional, Tuple, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import FakeData
from torchvision.transforms import transforms


__all__ = ['FakeDataModule']


class FakeDataModule(LightningDataModule):
    """
    Example of LightningDataModule for Fake dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        train_size: int = 320,
        test_size: int = 160,
        image_size: Tuple[int, int, int] = (3, 64, 64),
        num_classes: int = 10,
        data_dir: str = "data/",
        train_val_test_split: List = [160, 160, 160],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        is_customized: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.train_size = train_size
        self.test_size = test_size
        self.image_size = image_size
        self._num_classes = num_classes
        self.data_dir = data_dir
        if isinstance(train_val_test_split[0], float):
            total_num = train_size + test_size
            train_val_test_split = [int(total_num * ratio) for ratio in train_val_test_split]
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.is_customized = is_customized

        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

        # self.dims is returned when you call datamodule.size()
        self.dims = self.image_size

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = FakeData(size=self.train_size, image_size=self.image_size, transform=self.transforms)
        testset = FakeData(size=self.test_size, image_size=self.image_size, transform=self.transforms)
        dataset = ConcatDataset(datasets=[trainset, testset])
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split
        )

    def train_dataloader(self):
        train_loader = DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )
        if self.is_customized:
            train_val_loader = {
                'train': train_loader,
                'val': self.val_dataloader()
            }
            return train_val_loader
        return train_loader

    def val_dataloader(self):
        if self.is_customized:
            return self.test_dataloader()
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
