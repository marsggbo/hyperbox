import itertools
import os
import random
import types
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from hyperbox.datamodules import CIFAR10DataModule
from hyperbox_app.feature_shift.bnnas import BNNet
from hyperbox_app.feature_shift.nasbench201 import NASBench201Network
from hyperbox_app.feature_shift.tvmodels import *
from hyperbox_app.feature_shift.twonn import estimate_id
from hyperbox_app.feature_shift.twonn2 import twonn_dimension


def plot_ids_by_layers(model_ids, figsize=(8,8), topk=None):
    '''
    Args:
    model_ids: dict. {
        0: {'acc': 0.9288, 'IDs': [13,14,18,35,45,32,16,12,5]},
        1: {'acc': 0.9365, 'IDs': [...]},
        ...
    }
    '''
    marker = itertools.cycle(('+', '<',  'd', 'h', 'H','1', '.', '2', 'D', 'o', '*', 'v', '>')) 
    fig = plt.figure(num=1,figsize=figsize)
    ax = fig.add_subplot(111)
    for key, value in model_ids.items():
        if topk is not None and key > topk:
            break
        label, IDs = value['label'], value['IDs']
        label = f"{key}_{label}"
        x_axis = list(range(len(IDs)))
        y_axis = IDs
        color = (random.random(), random.random(), random.random())
        # print(x_axis, y_axis)
        ax.plot(x_axis, y_axis, color=color, marker=next(marker), label=label)
    ax.legend()
    plt.savefig('model_ids.pdf')
    plt.show()

def load_state_dict(net, ckptfile):
    ckpt = torch.load(ckptfile, map_location='cpu')
    weights = {}
    for key, value in ckpt['state_dict'].items():
        if 'network' in key:
            weights[key.replace('network.', '')] = value
    net.load_state_dict(weights)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='~/datasets/cifar10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=True, num_workers=4)

    for batch_idx, (imgs, labels) in enumerate(testloader):
        if batch_idx==2:
            imgs = imgs.to(device)
            labels = labels.to(device)
            print(labels)
            batch_size = imgs.shape[0]
            break
    print('Dataset done')

    # masks = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_all20_bn20/bnnas_c10_all20_bn20_search/2021-10-27_05-33-45/pareto_json/*.json')[:10]
    # ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_all20_bn20/*_nodepth_finetune*/*/checkpoints/*/*.ckpt')
    # # masks = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas/bnnas_random/*.json')[:10]
    # # ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_random_nodepth/*/*/checkpoints/*/*.ckpt')
    masks = glob('/home/xihe/xinhe/hyperbox/logs/runs/nasbench201_finetune/masks/nasbench201_mask_*.json')
    ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/nasbench201_finetune/nasbench201_finetune*/*/checkpoints/*/*.ckpt')
    ckpts = sorted(ckpts)
    masks = sorted(masks)
    getAcc = lambda ckpt: ckpt.split('acc=')[1].split('.ckpt')[0]
    accs = [float(getAcc(ckpt)) for ckpt in ckpts]
    indices = np.argsort(accs)[::-1]
    print(np.array(accs)[indices])
    ckpts = np.array(ckpts)[indices]
    masks = np.array(masks)[indices]
    ckpts[:2], masks[:2]
    print('ckpts and masks done')

    # ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/tvmodels_finetune/*/*/checkpoints/*/*.ckpt')
    
    networks = {}
    features = {}
    for idx, ckpt in enumerate(ckpts):
        mask = masks[idx]
        # BN-NAS
        # net = BNNet(num_classes=10, mask=mask, search_depth=False).to(device)
        # model_name = 'bnnas'

        # NAS-Bench-201
        net = NASBench201Network(mask=masks[idx]).to(device)
        model_name = 'nas201'

        # torchvision models
        # model_name = ckpt.split('finetune_')[1].split('/')[0]
        # net = eval(f"torchvision.models.{model_name}(num_classes=10)").to(device)
        if 'vgg' in model_name:
            net.forward = types.MethodType(vgg_forward, net)
        elif 'resnet' in model_name:
            net.forward = types.MethodType(resent_forward, net)
        elif 'mobile' in model_name:
            net.forward = types.MethodType(mbv3_forward, net)
        elif 'mnas' in model_name:
            net.forward = types.MethodType(mnas_forward, net)
        elif 'dense' in model_name:
            net.forward = types.MethodType(densenet_forward, net)
        elif 'squeeze' in model_name:
            net.forward = types.MethodType(squeeze_forward, net)
        else:
            pass

        load_state_dict(net, ckpt)
        net.eval()
        # networks[idx] = net
        y = net(imgs)
        features[idx] = net.features_list
    print('Features done')

    model_ids = {}
    for key, feats in features.items():
        try:
            acc = float(ckpts[key].split('acc=')[1].split('.ckpt')[0])
            # model_name = 'bnnas'
            model_name = 'nas201'
            # model_name = ckpts[key].split('finetune_')[1].split('/')[0]
            model_ids[key] = {'label': f"{model_name}_{acc}", 'IDs': []}
            for i, feat in enumerate(feats):
                # _id = estimate_id(feat.detach().cpu().numpy(), plot=False)    # the first method to calcutate ID value
                _id = twonn_dimension(feat.detach().cpu().numpy())              # the second method to calcutate ID value
                model_ids[key]['IDs'].append(_id)
        except Exception as e:
            print(e)
            # print(key, i)
    print('ID done')

    plot_ids_by_layers(model_ids, (8, 8), 10)
    print('done')        
