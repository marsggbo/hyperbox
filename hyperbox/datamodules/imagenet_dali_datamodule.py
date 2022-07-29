'''
https://github1s.com/PyTorchLightning/pytorch-lightning/blob/HEAD/pl_examples/basic_examples/dali_image_classifier.py
'''
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from nvidia.dali import __version__ as dali_version
from nvidia.dali import fn, ops, types
from nvidia.dali.pipeline import Pipeline, pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from packaging.version import Version
from pytorch_lightning import LightningDataModule


NEW_DALI_API = Version(dali_version) >= Version('0.28.0')
if NEW_DALI_API:
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy


__all__ = ['ImagenetDALIDataModule', 'create_dali_pipeline']

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


class ImagenetDALIDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_path: str,
        validation_size: Optional[Union[int, float]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        dali_cpu: bool = False,
        val_size: int = 256,
        crop_size: int = 224
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.dali_cpu = dali_cpu
        self.val_size = val_size
        self.crop_size = crop_size

    def cat_path(self, path_list):
        return os.path.join(*path_list)

    @property
    def global_rank(self):
        if getattr(self, 'trainer', None) is not None:
            return self.trainer.global_rank
        return 0

    @property
    def local_rank(self):
        if getattr(self, 'trainer', None) is not None:
            return self.trainer.local_rank
        return 0

    @property
    def world_size(self):
        if getattr(self, 'trainer', None) is not None:
            return self.trainer.world_size
        return 0

    def setup(self, stage: Optional[str]=None):
        self.train_pipe = create_dali_pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_workers,
            num_shards=self.world_size,
            shard_id=self.global_rank,
            device_id=self.local_rank,
            seed=666+self.global_rank,
            data_dir=self.cat_path([self.dataset_path, 'train']),
            dali_cpu=self.dali_cpu,
            crop=self.crop_size,
            size=self.val_size,
            is_training=True
        )
        self.train_pipe.build()

        self.val_pipe = create_dali_pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_workers,
            num_shards=self.world_size,
            shard_id=self.global_rank,
            device_id=self.local_rank,
            seed=666+self.global_rank,
            data_dir=self.cat_path([self.dataset_path, 'val']),
            dali_cpu=self.dali_cpu,
            crop=self.crop_size,
            size=self.val_size,
            is_training=False
        )
        self.val_pipe.build()

    def train_dataloader(self):
        return DALIClassificationIterator(
            self.train_pipe,
            reader_name="Reader",
            auto_reset=True,
            fill_last_batch=True,
            last_batch_policy=LastBatchPolicy.PARTIAL
        )

    def val_dataloader(self):
        return DALIClassificationIterator(
            self.val_pipe,
            reader_name="Reader",
            auto_reset=True,
            fill_last_batch=False,
            last_batch_policy=LastBatchPolicy.PARTIAL
        )

    def test_dataloader(self):
        return DALIClassificationIterator(
            self.val_pipe,
            reader_name="Reader",
            auto_reset=True,
            fill_last_batch=False,
            last_batch_policy=LastBatchPolicy.PARTIAL
        )


class ExternalImageNetInputIterator(object):
    """
    This iterator class wraps torchvision's ImageNet dataset and returns the images and labels in batches
    """

    def __init__(self, imagenet_ds, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.imagenet_ds = imagenet_ds
        self.indices = list(range(len(self.imagenet_ds)))
        if shuffle: shuffle(self.indices)

    def __iter__(self):
        self.i = 0
        self.n = len(self.imagenet_ds)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            index = self.indices[self.i]
            img, label = self.imagenet_ds[index]
            batch.append(img.numpy())
            labels.append(np.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)


class ExternalSourcePipeline(Pipeline):
    """
    This DALI pipeline class just contains the ImageNet iterator
    """

    def __init__(self, batch_size, eii, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.source = ops.ExternalSource(source=eii, num_outputs=2)
        self.build()

    def define_graph(self):
        images, labels = self.source()
        return images, labels


class DALIClassificationLoader(DALIClassificationIterator):
    """
    This class extends DALI's original `DALIClassificationIterator` with the `__len__()` function
     so that we can call `len()` on it
    """

    def __init__(
        self,
        pipelines,
        size=-1,
        reader_name=None,
        auto_reset=False,
        fill_last_batch=True,
        dynamic_shape=False,
        last_batch_padded=False,
    ):
        if NEW_DALI_API:
            last_batch_policy = LastBatchPolicy.FILL if fill_last_batch else LastBatchPolicy.DROP
            super().__init__(
                pipelines,
                size,
                reader_name,
                auto_reset,
                dynamic_shape,
                last_batch_policy=last_batch_policy,
                last_batch_padded=last_batch_padded
            )
        else:
            super().__init__(
                pipelines, size, reader_name, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded
            )
        self._fill_last_batch = fill_last_batch

    def __len__(self):
        batch_count = self._size // (self._num_gpus * self.batch_size)
        last_batch = 1 if self._fill_last_batch else 1
        return batch_count + last_batch



if __name__ == '__main__':
    dm = ImagenetDALIDataModule('~/datasets/imagenet2012')
    dm.setup()
    for i, data in enumerate(dm.train_dataloader()):
        if i > 2: break
        img, label = data[0]['data'], data[0]['label']
        print(f"Train {i} label={label.view(-1)}")
    for i, data in enumerate(dm.val_dataloader()):
        if i > 2: break
        img, label = data[0]['data'], data[0]['label']
        print(f"Val {i} label={label.view(-1)}")
    for i, data in enumerate(dm.test_dataloader()):
        if i > 2: break
        img, label = data[0]['data'], data[0]['label']
        print(f"Test {i} label={label.view(-1)}")
    pass
