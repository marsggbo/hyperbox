
from typing import Union, Optional

from omegaconf import DictConfig
from albumentations import (CLAHE, Blur, ChannelDropout, ChannelShuffle,
                            Compose, Cutout, Flip, GaussNoise, GridDistortion,
                            HueSaturationValue, IAAAdditiveGaussianNoise,
                            IAAEmboss, MotionBlur, Normalize, OneOf,
                            OpticalDistortion, RandomBrightnessContrast,
                            RandomGridShuffle, Resize, ShiftScaleRotate)
from albumentations.pytorch.transforms import ToTensor

from .base_transforms import BaseTransforms

__all__ = [
    'AlbumentationsTransforms'
]


class AlbumentationsTransforms(BaseTransforms):
    def __init__(
        self,
        input_size: Union[list],
        random_grid_shuffle: Union[dict, DictConfig] = {'enable': 0, 'grid': 2},
        channel_shuffle: Union[dict, DictConfig] = {'enable': 0, 'p': 1},
        channel_dropout: Union[dict, DictConfig] = {
            'enable': 0, 'drop_range': (1,1), 'fill_value': 127, 'p': 1},
        noise: Union[dict, DictConfig] = {'enable': 0},
        blur: Union[dict, DictConfig] = {'enable': 0},
        rotate: Union[dict, DictConfig] = {'enable': 0},
        distortion: Union[dict, DictConfig] = {'enable': 0},
        bright: Union[dict, DictConfig] = {'enable': 0},
        hue: Union[dict, DictConfig] = {'enable': 0},
        cutout: Union[dict, DictConfig] = {
            'enable': 0, 'num_holes': 10, 'size': 20, 'fill_value': 127},
        to_tensor: Union[dict, DictConfig] = {'enable': 1},
        normalize: Union[dict, DictConfig] = {
            'enable': 1, 'mean': [0.5,0.5,0.5], 'std': [0.5,0.5,0.5]},
    ):
        super(AlbumentationsTransforms, self).__init__()
        self._transform_train = self.parse_transforms()
        self._transform_valid = [Resize(*self.input_size)]
        if self.normalize.enable:
            mean = self.normalize.mean
            std = self.normalize.std
            self.normalize = Normalize(mean, std)
            self._transform_valid.append(self.normalize)
        if self.to_tensor.enable:
            self._transform_valid.append(ToTensor())
    
    def parse_transforms(self, image):
        height, width = self.input_size
        transforms_list = []
        Resize(height, width), Flip()
        # random_grid_shuffle
        if self.random_grid_shuffle.enable:
            grid = self.random_grid_shuffle.grid
            grid = (grid,grid)
            transforms_list.append(RandomGridShuffle((grid)))

        # channel_shuffle
        if self.channel_shuffle.enable:
            p = self.channel_shuffle.p
            transforms_list.append(ChannelShuffle(p=p))
        
        # channel_dropout
        if self.channel_dropout.enable:
            drop_range = self.channel_dropout.drop_range
            fill_value = self.channel_dropout.fill_value
            p = self.channel_dropout.p
            transforms_list.append(ChannelDropout(drop_range, fill_value, p=p))

        # noise
        if self.noise.enable:
            transforms_list.append(OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=1))

        # blur
        if self.blur.enable:
            transforms_list.append(OneOf([
                MotionBlur(),
                Blur(blur_limit=3,),
            ], p=1))

        # rotate
        if self.rotate.enable:
            transforms_list.append(
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                                                    rotate_limit=45, p=1))

        # distortion
        if self.distortion.enable:
            transforms_list.append(OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.3),
            ], p=1))

        # bright
        if self.bright.enable:
            transforms_list.append(
                OneOf([
                    CLAHE(clip_limit=2),
                    RandomBrightnessContrast(p=0.8),            
                ], p=1))

        # hue color
        if self.hue.enable:
            transforms_list.append(HueSaturationValue(p=0.3))
        
        # cutout
        if self.cutout.enable:
            num_holes = self.cutout.num_holes
            size = self.cutout.size
            fill_value = self.cutout.fill_value
            transforms_list.append(Cutout(num_holes, size, size, fill_value, 1))
        if self.normalize.enable:
            mean = self.normalize.mean
            std = self.normalize.std
            self.normalize = Normalize(mean, std)
            transform_list.append()
        if self.to_tensor.enable:
            transform_list.append(ToTensor())

        return Compose(transforms_list)

    @property
    def valid_transform(self):
        return self._transform_valid

    @property
    def train_transform(self):
        return self._transform_train
