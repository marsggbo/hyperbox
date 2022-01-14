from typing import Optional, Tuple, Union, Any, List, Callable

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from pl_bolts.datamodules import CIFAR10DataModule as bolt_cifar10
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms

from hyperbox.datamodules.transforms import get_transforms
from hyperbox.datamodules.transforms.cutout import Cutout


__all__ = ['CIFAR10DataModule', 'CIFAR100DataModule']


class CIFAR10DataModule(bolt_cifar10):
    """
    Example of LightningDataModule for CIFAR10 dataset.

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
    name = "cifar10"
    dataset_cls = CIFAR10
    dims = (3, 32, 32)
    EXTRA_ARGS = {}
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]

    def __init__(
        self,
        transforms: Union[dict, DictConfig]={},
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.5,
        num_workers: int = 4,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 666,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        num_classes: int = 10,
        is_customized: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            data_dir, val_split, num_workers, normalize, batch_size, seed,
            shuffle, pin_memory, drop_last, *args, **kwargs)
        self._transforms = get_transforms('torch', dict(transforms))
        self._num_classes = num_classes
        self.is_customized = is_customized

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def default_train_transforms(self):
        # return self._transforms._transform_train
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            Cutout(1, 16),
            transforms.Normalize(self.MEAN, self.STD)
        ])

    def default_transforms(self) -> Callable:
        """ Default transform for the dataset """
        # return self._transforms._transform_valid
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD)
            ]
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = self.default_train_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )

            # Split
            if self.is_customized:
                dataset_to_split = self.dataset_cls(self.data_dir, train=True, transform=val_transforms, **self.EXTRA_ARGS)
                self.dataset_train = self._split_dataset(dataset_to_split, train=True)
                self.dataset_val = self._split_dataset(dataset_to_split, train=False)
            else:
                self.dataset_train = self.dataset_cls(self.data_dir, train=True, transform=train_transforms, **self.EXTRA_ARGS)
                self.dataset_val = self.dataset_test

        elif stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )

    def train_dataloader(self):
        train_loader = self._data_loader(self.dataset_train, shuffle=self.shuffle)
        if self.is_customized:
            val_loader = self._data_loader(self.dataset_val, shuffle=self.shuffle)
            train_val_loader = {
                'train': train_loader,
                'val': val_loader
            }
            return train_val_loader
        return train_loader

    def val_dataloader(self):
        if self.is_customized:
            return self.test_dataloader()
        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        return self._data_loader(self.dataset_test)


class CIFAR100DataModule(CIFAR10DataModule):
    
    name = "cifar100"
    dataset_cls = CIFAR100
    MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    def __init__(
        self,
        transforms: Union[dict, DictConfig],
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.5,
        num_workers: int = 4,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 666,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        num_classes: int = 100,
        is_customized: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(transforms, data_dir, val_split, num_workers, normalize, batch_size, seed,
                         shuffle, pin_memory, drop_last, num_classes, is_customized, *args, **kwargs)
        