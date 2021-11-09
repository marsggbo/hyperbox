
from typing import Union, Optional

from omegaconf import DictConfig
from torchvision import transforms

from .cutout import Cutout
from .base_transforms import BaseTransforms


class TorchTransforms(BaseTransforms):
    def __init__(
        self,
        input_size: Union[list] = [32],
        random_resized_crop: Union[dict, DictConfig] = {'enable': 0, 'padding': 0},
        resize: Union[dict, DictConfig] = {'enable': 0},
        random_crop: Union[dict, DictConfig] = {'enable': 0},
        center_crop: Union[dict, DictConfig] = {'enable': 0},
        color_jitter: Union[dict, DictConfig] = {'enable': 0},
        random_horizontal_flip: Union[dict, DictConfig] = {'enable': 0, 'p': 0.5},
        random_vertical_flip: Union[dict, DictConfig] = {'enable': 0, 'p': 0.5},
        random_rotation: Union[dict, DictConfig] = {'enable': 0, 'degrees': 20},
        cutout: Union[dict, DictConfig] = {'enable': 0, 'n_holes': 8, 'length': 4},
        to_tensor: Union[dict, DictConfig] = {'enable': 1},
        normalize: Union[dict, DictConfig] = {
            'enable': 1, 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    ):
        super(TorchTransforms, self).__init__()
        self.min_edge_size = min(input_size)
        self._transform_train = self.parse_transforms()

        self._transform_valid = [
            transforms.Resize(self.input_size)
        ]
        if self.to_tensor.enable:
            self._transform_valid.append(transforms.ToTensor())
        if self.normalize.enable:
            mean = self.normalize.mean
            std = self.normalize.std
            self.normalize = transforms.Normalize(mean, std)
            self._transform_valid.append(self.normalize)
        self._transform_valid = transforms.Compose(self._transform_valid)

    def parse_transforms(self):
        img_size = self.input_size
        transform_list = []

        # resize and crop opertaion
        if self.random_resized_crop.enable:
            transform_list.append(transforms.RandomResizedCrop(img_size))
        elif self.resize.enable:
            transform_list.append(transforms.Resize(img_size))
        if self.random_crop.enable:
            padding = self.random_crop.padding
            size = getattr(self.random_crop, 'size', self.min_edge_size)
            transform_list.append(transforms.RandomCrop(size, padding=padding))
        elif self.center_crop.enable:
            transform_list.append(transforms.CenterCrop(self.min_edge_size))

        # ColorJitter
        if self.color_jitter.enable:
            params = {key: self.color_jitter[key] for key in self.color_jitter
                      if key != 'enable'}
            transform_list.append(transforms.ColorJitter(**params))

        # horizontal flip
        if self.random_horizontal_flip.enable:
            p = self.random_horizontal_flip.p
            transform_list.append(transforms.RandomHorizontalFlip(p))

        # vertical flip
        if self.random_vertical_flip.enable:
            p = self.random_vertical_flip.p
            transform_list.append(transforms.RandomVerticalFlip(p))

        # rotation
        if self.random_rotation.enable:
            degrees = self.random_rotation.degrees
            transform_list.append(transforms.RandomRotation(degrees))

        # cutout
        if self.to_tensor.enable:
            transform_list.append(transforms.ToTensor())
        if self.cutout.enable:
            n_holes = self.cutout.n_holes
            length = self.cutout.length
            transform_list.append(Cutout(n_holes, length))
        if self.normalize.enable:
            mean = self.normalize.mean
            std = self.normalize.std
            transform_list.append(transforms.Normalize(mean, std))
        transform_list = transforms.Compose(transform_list)
        assert len(transform_list.transforms) > 0, "The length of transform list much be larger than 0."
        return transform_list

    @property
    def transform_train(self):
        raise self._transform_train

    @property
    def transform_valid(self):
        return self._transform_train
