from typing import Optional, Tuple

import medmnist
from medmnist import INFO

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

__all__ = ['MedMNISTDataModule']


np2tensor = lambda x: torch.from_numpy(x).float()
squeeze_tensor = lambda x: torch.from_numpy(x).squeeze().long()


class MedMNISTDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MedMNIST dataset.
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
        data_dir: str,
        data_flag: str,
        num_classes: int=None,
        download: bool = True,
        batch_size: int = 64,
        num_workers: int = 4,
        train_transform = None,
        val_transform = None,
        pin_memory: bool = False,
        is_customized: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.data_flag = data_flag
        self.batch_size = batch_size
        self.info = INFO[data_flag]
        self.num_classes = len(self.info['label']) if num_classes is None else num_classes
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.is_customized = is_customized

        if '3d' in data_flag.lower():
            self.is_3d = True
            to_tensor = np2tensor
            default_train_transform = default_val_transform = transforms.Compose(
                [to_tensor]
            )
        else:
            self.is_3d = False
            to_tensor = transforms.ToTensor()
            default_train_transform = default_val_transform = transforms.Compose(
                [to_tensor, transforms.Normalize((0.5,), (0.5,))]
            )

        if train_transform is None:
            self.train_transform = default_train_transform
        else:
            self.train_transform = train_transform
        if val_transform is None:
            self.val_transform = default_val_transform
        else:
            self.val_transform = val_transform

        if self.info['task'] != 'multi-label, binary-class':
            self.target_transform = squeeze_tensor

        # self.dims is returned when you call datamodule.size()
        self.dims = (self.info['n_channels'], 28, 28)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        DataClass = getattr(medmnist, self.info['python_class'])
        DataClass(split='train', root=self.data_dir, download=True)
        DataClass(split='test', root=self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        DataClass = getattr(medmnist, self.info['python_class'])
        self.data_train = DataClass(
            split='train', transform=self.train_transform, target_transform=self.target_transform,
            root=self.data_dir, download=True)
        self.data_val = DataClass(
            split='val', transform=self.val_transform, target_transform=self.target_transform,
            root=self.data_dir, download=True)
        self.data_test = DataClass(
            split='test', transform=self.val_transform, target_transform=self.target_transform,
            root=self.data_dir, download=True)

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


if __name__ == '__main__':
    data_dir = '/home/datasets/Flythings3D/medmnist/'
    data_flag = 'synapsemnist3d'
    # data_flag = 'chestmnist'
    # data_flag = 'bloodmnist'
    mmd = MedMNISTDataModule(data_dir=data_dir, data_flag=data_flag)
    mmd.setup()
    for idx, (imgs, labels) in enumerate(mmd.train_dataloader()):
        if idx <= 2:
            if mmd.data_train.info['task']=='multi-label, binary-class':
                labels = labels.to(torch.float32)
            else:
                labels = labels.squeeze().long()
            print(imgs.shape, labels.shape)
            # 3D case: torch.Size([64, 1, 28, 28, 28]) torch.Size([64, 1])
            # 2D case: torch.Size([64, 1, 28, 28]) torch.Size([64, 1])
        else:
            break