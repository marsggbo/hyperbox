
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from pl_bolts.datamodules import ImagenetDataModule as bolt_imagenet
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, ImageNet
from torchvision.transforms import transforms as transform_lib


__all__ = ['ImagenetDataModule']


class ImagenetDataModule(bolt_imagenet):
    def __init__(
        self,
        data_dir: str,
        meta_dir: Optional[str] = None,
        classes: int = 1000,
        image_size: Optional[Union[int, List]] = 224,
        valid_image_size: Optional[Union[int, List]] = None, # valid is the test set by default
        num_imgs_per_val_class: int = 50,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        autoaugment: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._num_classes = classes
        self.valid_image_size = valid_image_size
        self.autoaugment = autoaugment
        if '~' in data_dir:
            data_dir = os.path.expanduser(data_dir)
        super(ImagenetDataModule, self).__init__(
            data_dir, meta_dir, num_imgs_per_val_class, image_size, num_workers,
            batch_size, shuffle, pin_memory, drop_last, *args, **kwargs)

    def prepare_data(self) -> None:
        pass

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def train_transform(self, size=None) -> Callable:
        if size is None:
            size = self.image_size
        op_list = [
            transform_lib.RandomResizedCrop(size),
            transform_lib.RandomHorizontalFlip(),            
        ]
        if self.autoaugment:
            policy = T.AutoAugmentPolicy.IMAGENET
            augmenter = T.AutoAugment(policy)
            op_list.append(augmenter)
        else:
            op_list.append(
                transform_lib.ColorJitter(brightness=32. / 255., saturation=0.5),
            )
        op_list.extend([
            transform_lib.ToTensor(),
            transform_lib.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )  
        ])
        
        preprocessing = transform_lib.Compose(op_list)
        return preprocessing

    def val_transform(self) -> Callable:
        preprocessing = transform_lib.Compose([
            transform_lib.Resize(self.image_size + 32),
            transform_lib.CenterCrop(self.image_size),
            transform_lib.ToTensor(),
            transform_lib.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return preprocessing

    def setup(self, stage: Optional[str] = None):
        if stage in ['fit', 'train'] or stage is None:
            transform_train = self.train_transform() if self.train_transforms is None else self.train_transforms
            self.data_train = ImageFolder(
                root=os.path.join(self.data_dir, 'train'),
                transform=transform_train
            )

        if stage in ['fit', 'test'] or stage is None:
            transform_test = self.val_transform() if self.test_transforms is None else self.test_transforms
            self.data_test = ImageFolder(
                root=os.path.join(self.data_dir, 'val'),
                transform=transform_test
            )

        if stage is None or stage=='fit':
            if self.valid_image_size is not None:
                transform_valid = self.train_transform(self.valid_image_size)
                validation_size = self.num_imgs_per_val_class * self.num_classes
                train_size = len(self.data_train) - validation_size
                self.data_train, self.data_valid = random_split(self.data_train, [train_size, validation_size])
            else:
                self.data_valid = self.data_test
                self.data_train = self.data_train

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

