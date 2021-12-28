from typing import Union, List, Optional
import json
import os
import random
import nibabel as nib

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as TF

from torch.utils.data import DataLoader, Dataset, random_split
from kornia import image_to_tensor, tensor_to_image
from pytorch_lightning import LightningDataModule
from hyperbox.utils.utils import hparams_wrapper
from hyperbox_app.medmnist.datamodules.utils import RandomResampler, SymmetricalResampler, pil_loader


__all__ = [
    'CTDataset',
    'CTDatamodule'
]


@hparams_wrapper
class CTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        data_list: str,
        is_train: bool,
        is_color: bool=False,
        is_3d: bool=True,
        img_size: Union[List, int]=[512,512],
        center_size: Union[List, int]=[360,360],
        slice_num: int=64,
        loader=pil_loader,
        transforms=None,
        label_transforms=None,
        *args, **kwargs
    ):
        '''
        Args:
            root_dir: root dir of dataset, e.g., ~/../../datasets/CCCCI_cleaned/dataset_cleaned/
            data_list: the training of testing data list or json file. e.g., ct_train.json
            is_train: determine to load which type of dataset
            slice_num: the number of slices in a scan
        '''
        with open(self.data_list, 'r') as f:
            self.data = json.load(f)
        self.cls_to_label = {
            # png slices
            'CP': 0, 'NCP': 1, 'Normal': 2,
            # nii
            'CT-0': 0, 'CT-1': 1, 'CT-2': 1, 'CT-3': 1, 'CT-4': 1,
            # covid_ctset
            'normal': 0, 'covid': 1
        }
        self.samples = self.convert_json_to_list(self.data)

    def convert_json_to_list(self, data):
        samples = {} # {0: {'scans': [], 'labels': 0}}
        idx = 0
        for cls_ in data:
            for pid in data[cls_]:
                for scan_id in data[cls_][pid]:
                    slices = data[cls_][pid][scan_id]
                    label = self.cls_to_label[cls_]
                    if slices[0].endswith('.nii') or slices[0].endswith('.gz'):
                        scan_path = os.path.join(self.root_dir,cls_,slices[0])
                    else:
                        scan_path = os.path.join(self.root_dir,cls_,pid,scan_id)
                    if os.path.exists(scan_path) and len(slices)>0:
                            samples[idx] = {'slices':slices, 'label': label, 'path': scan_path}
                            idx += 1
        return samples

    def preprocessing(self, img):
        # resize = int(self.img_size[0]*5/4)
        resize = int(self.img_size[0])
        transform = TF.Compose([
            TF.Resize(self.img_size),
            TF.CenterCrop(self.center_size),
            TF.ToTensor()
        ])
        return transform(img)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = torch.tensor(sample['label']).long()
        # stack & sample slice
        if sample['slices'][0].endswith('.nii') or sample['slices'][0].endswith('.nii.gz'):
            slice_tensor = self.get_nifti(sample)
        else:
            slice_tensor = self.get_png(sample)

        # if not 3d, then remove channel dimension
        if not self.is_3d: slice_tensor = slice_tensor[0, :, :, :]
        return slice_tensor, label
        # return slice_tensor, label, sample['path']

    def get_nifti(self, sample):
        path = sample['path']
        slice_tensor = []
        slice_path = path
        img = nib.load(slice_path) 
        img_fdata = img.get_fdata()
        (x,y,z) = img.shape
        slice_tensor = torch.FloatTensor(img_fdata)
        slice_tensor = slice_tensor.unsqueeze(dim=0)
        slice_tensor = slice_tensor.permute(0, 3, 1, 2)
        if self.is_train:
            slices = RandomResampler.resample(list(range(z)), self.slice_num)
        else:
            slices = SymmetricalResampler.resample(list(range(z)), self.slice_num)
        slice_tensor = slice_tensor[:, slices, :, :]
        # todo: imbalanced problem
        h, w = self.img_size[0], self.img_size[1]
        size = (h*5//4, w*5//4)
        slice_tensor = torch.nn.functional.interpolate(slice_tensor, size) # resize
        slice_tensor = slice_tensor[:, :, size[0]-h//2:size[0]+h//2, size[1]-w//2:size[1]+w//2] # centercrop

        return slice_tensor

    def get_png(self, sample):
        path = sample['path']
        if self.is_train:
            slices = RandomResampler.resample(sample['slices'], self.slice_num)
        else:
            slices = SymmetricalResampler.resample(sample['slices'], self.slice_num)

        slice_tensor = []
        for slice_ in slices:
            slice_path = os.path.join(path, slice_)
            img = self.loader(slice_path) # height * width
            img = self.preprocessing(img) # 1 * height * width
            if not self.is_color:
                img = torch.unsqueeze(img[0, :, :], dim=0)
            slice_tensor.append(img)
        slice_tensor = torch.stack(slice_tensor)
        slice_tensor = slice_tensor.permute(1, 0, 2, 3) # c*d*h*w

        return slice_tensor

    def __len__(self):
        return len(self.samples)


@hparams_wrapper
class CTDatamodule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        data_list_train: str,
        data_list_val: str,
        data_list_test: str,
        is_color: bool=True,
        is_3d: bool=True,
        img_size: Union[List, int]=[512, 512],
        center_size: Union[List, int]=[360, 360],
        batch_size: int=16,
        slice_num: int=64,
        seed: int = 666,
        is_customized: bool = False,
        num_workers: int = 4,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        self.is_setup = False
        self.setup()

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return
        self.dataset_train = CTDataset(
            self.root_dir, data_list=self.data_list_train, is_train=True, is_color=self.is_color,
            is_3d=self.is_3d, img_size=self.img_size, center_size=self.center_size, slice_num=self.slice_num)
        self.dataset_val = CTDataset(
            self.root_dir, data_list=self.data_list_val, is_train=True, is_color=self.is_color,
            is_3d=self.is_3d, img_size=self.img_size, center_size=self.center_size, slice_num=self.slice_num)
        self.dataset_test = CTDataset(
            self.root_dir, data_list=self.data_list_test, is_train=True, is_color=self.is_color,
            is_3d=self.is_3d, img_size=self.img_size, center_size=self.center_size, slice_num=self.slice_num)
        self.datasets = [
            self.dataset_train, self.dataset_val, self.dataset_test
        ]
        self.is_setup = True

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        train_loader = self._data_loader(self.dataset_train, shuffle=True)
        if self.is_customized:
            train_val_loader = {
                'train': train_loader,
                'val': self.val_dataloader()
            }
            return train_val_loader
        return train_loader

    def val_dataloader(self):
        return self.test_dataloader()
        if self.is_customized:
            return self.test_dataloader()
        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        return self._data_loader(self.dataset_test)

    def change_img_size(self, img_size):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        for dataset in self.datasets:
            dataset.img_size = img_size

    def change_slice_num(self, slice_num):
        for dataset in self.datasets:
            dataset.slice_num = slice_num


if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.INFO)
    # dataset_cccci = CTDataset(
    #     root_dir='/home/datasets/CCCCI_cleaned/dataset_cleaned/',
    #     data_list='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/ccccii/ct_train.json',
    #     is_train=True
    # )
    # dataset_nii = CTDataset(
    #     root_dir='/home/datasets/MosMedData/COVID19_1110/pngs',
    #     data_list='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/mosmed/nii_png_train.json',
    #     is_train=True
    # )
    # dataset_ct = CTDataset(
    #     root_dir='/home/datasets/COVID-CTset_visual',
    #     data_list='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datasets/covid_ctset/train_balance.json',
    #     is_train=True
    # )
    # for dataset in [dataset_cccci, dataset_nii, dataset_ct]:
    #     data = dataset[0]
    #     logging.info(data[0].shape)

    # datamodule = CTDatamodule(
    #     root_dir='/home/datasets/CCCCI_cleaned/dataset_cleaned/',
    #     is_color=False,
    #     num_workers=0,
    #     data_list_train='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datamodules/ccccii/ct_train.json',
    #     data_list_val='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datamodules/ccccii/ct_test.json',
    #     data_list_test='/home/comp/18481086/code/hyperbox/hyperbox_app/covid19/datamodules/ccccii/ct_test.json',
    # )
    print('start')
    logging.info('start')
    datamodule = CTDatamodule(
        root_dir='/home/datasets/COVID-CTset_visual',
        is_color=False,
        num_workers=0,
        data_list_train='/home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/datamodules/iran/train.json',
        data_list_val='/home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/datamodules/iran/test.json',
        data_list_test='/home/comp/18481086/code/hyperbox/hyperbox_app/medmnist/datamodules/iran/test.json',
    )
    logging.info('dataset done')
    from time import time
    start = time()
    for idx, data in enumerate(datamodule.train_dataloader()):
        if idx > 20:
            break
        img, label = data
        logging.info(img.shape)
    cost = time() - start
    logging.info(f"cost {cost/(idx+1)} sec")
    logging.info('end')