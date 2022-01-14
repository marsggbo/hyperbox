import os
from typing import List, Optional, Tuple

import medmnist
import numpy as np
import torch
from medmnist import INFO
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import transforms

from hyperbox.datamodules.distributed_sampler_wrapper import DistributedSamplerWrapper

__all__ = ['MedMNISTDataModule']


np2tensor = lambda x: torch.from_numpy(x).float()
squeeze_tensor = lambda x: torch.from_numpy(x).squeeze().long()


class ShapeTransform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
   
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        
        return voxel.astype(np.float32)


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
        as_rgb: bool = False,
        num_classes: int=None,
        download: bool = True,
        batch_size: int = 64,
        num_workers: int = 4,
        train_transform = None,
        val_transform = None,
        pin_memory: bool = False,
        shape_transform: bool = False,
        is_customized: bool = False,
        concat_train_val: bool = False,
        use_balanced_batch_sampler: bool = False,
        use_weighted_sampler: bool = True,
        class_weights: List = None,
        **kwargs,
    ):
        super().__init__()
        if '~' in data_dir:
            data_dir = os.path.expanduser(data_dir)
        self.data_dir = data_dir
        self.data_flag = data_flag
        self.as_rgb = as_rgb
        self.batch_size = batch_size
        self.info = INFO[data_flag]
        self.num_classes = len(self.info['label']) if num_classes is None else num_classes
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shape_transform = shape_transform
        self.is_customized = is_customized
        self.use_balanced_batch_sampler = use_balanced_batch_sampler
        self.use_weighted_sampler = use_weighted_sampler
        self.concat_train_val = concat_train_val
        self.class_weights = class_weights

        if '3d' in data_flag.lower():
            self.is_3d = True
            to_tensor = np2tensor
            default_train_transform = transforms.Compose(
                [
                    ShapeTransform3D('random') if self.shape_transform else ShapeTransform3D(),
                    to_tensor]
            )
            default_val_transform = transforms.Compose(
                [
                    ShapeTransform3D(0.5) if self.shape_transform else ShapeTransform3D(),
                    to_tensor]
            )
        else:
            self.is_3d = False
            to_tensor = transforms.ToTensor()
            default_train_transform = default_val_transform = transforms.Compose(
                [
                    to_tensor
                ]
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
        ...
        # DataClass = getattr(medmnist, self.info['python_class'])
        # DataClass(split='train', root=self.data_dir, download=True)
        # DataClass(split='test', root=self.data_dir, download=True)

    def build_weighted_sampler(self, dataset, weights: list=None):
        '''
        weights: class weights list
        '''
        if weights is None:
            labels = torch.tensor(dataset.labels).view(-1)
            class_sample_count = torch.tensor(
                [(labels == t).sum() for t in torch.unique(labels, sorted=True)])
            weights = 1. / class_sample_count.float()
            print(weights)
        samples_weight = torch.tensor([weights[t.item()] for t in labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=False)
        return sampler

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        DataClass = getattr(medmnist, self.info['python_class'])
        self.data_train = DataClass(
            split='train', transform=self.train_transform, target_transform=self.target_transform,
            root=self.data_dir, download=True, as_rgb=self.as_rgb)
        self.data_val = DataClass(
            split='val', transform=self.val_transform, target_transform=self.target_transform,
            root=self.data_dir, download=True, as_rgb=self.as_rgb)
        self.data_test = DataClass(
            split='test', transform=self.val_transform, target_transform=self.target_transform,
            root=self.data_dir, download=True, as_rgb=self.as_rgb)
        self.num_classes = len(set(self.data_train.labels.reshape(-1).tolist()))

    def train_dataloader(self):
        sampler = None
        shuffle = True
        dataset = self.data_train
        if self.concat_train_val:
            dataset = ConcatDataset((self.data_train, self.data_val))
            dataset.labels = np.vstack([data.labels for data in dataset.datasets])
            dataset.imgs = np.vstack([data.imgs for data in dataset.datasets])
        if self.use_balanced_batch_sampler:
            from catalyst.data import BalanceClassSampler, BatchBalanceClassSampler
            labels = self.data_train.labels.reshape(-1)
            num_classes = len(set(labels))
            num_samples = self.batch_size
            num_batches = max(100,  len(labels) // (num_classes * num_samples))
            sampler = BatchBalanceClassSampler(
                labels.tolist(), num_classes=num_classes, num_samples=num_samples, num_batches=num_batches)
            # sampler = DistributedSamplerWrapper(sampler)
            train_loader = DataLoader(
                dataset=dataset,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                batch_sampler=sampler,
            )
        elif self.use_weighted_sampler:
            weights = self.class_weights
            sampler = self.build_weighted_sampler(dataset, weights)
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=sampler
            )
        else:
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=sampler,
                shuffle=shuffle
            )
        if self.is_customized:
            if self.use_balanced_batch_sampler:
                from catalyst.data import BalanceClassSampler, BatchBalanceClassSampler
                labels = self.data_val.labels.reshape(-1)
                num_classes = len(set(labels))
                num_samples = self.batch_size
                num_batches = max(40,  len(labels) // (num_classes * num_samples))
                sampler = BatchBalanceClassSampler(
                    labels.tolist(), num_classes=num_classes, num_samples=num_samples, num_batches=num_batches)
                # sampler = DistributedSamplerWrapper(sampler)
                val_dataloader = DataLoader(
                    dataset=self.data_val,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    batch_sampler=sampler,
                )
            elif self.use_weighted_sampler:
                weights = self.class_weights
                labels = self.data_val.labels.reshape(-1)
                num_classes = len(set(labels))
                num_samples = self.batch_size
                num_batches = max(40,  len(labels) // (num_classes * num_samples))
                sampler = self.build_weighted_sampler(self.data_val, weights)
                val_dataloader = DataLoader(
                    dataset=self.data_val,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    sampler=sampler,
                )
            else:
                val_dataloader = DataLoader(
                    dataset=self.data_val,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )
            train_val_loader = {
                'train': train_loader,
                'val': val_dataloader
            }
            return train_val_loader
        return train_loader

    def val_dataloader(self):
        if self.is_customized or self.concat_train_val:
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
    data_dir = '~/datasets/medmnist/'
    data_flag = 'vesselmnist3d'
    # data_flag = 'synapsemnist3d'
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
