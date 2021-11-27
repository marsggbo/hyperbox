import os
from glob import glob

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from torchvision.utils import make_grid

from hyperbox.datamodules import CIFAR10DataModule
from hyperbox_app.feature_shift.bnnas import BNNet
from hyperbox_app.feature_shift.nasbench201 import NASBench201Network


def show_images(images, idx, cols = 2, titles = None, filename='test.png'):
    """Display a list of images in a single figure with matplotlib.
    https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Subnet (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    plt.title(f"{idx}")
    fig.savefig(f'{filename}')

def load_state_dict(net, ckptfile):
    ckpt = torch.load(ckptfile, map_location='cpu')
    weights = {}
    for key, value in ckpt['state_dict'].items():
        if 'network' in key:
            weights[key.replace('network.', '')] = value
    net.load_state_dict(weights)

def feat_distance(feat):
    distance = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(feat.cpu().detach().numpy(),
        metric='correlation'))
    # distance = torch.mm(feat, feat.T)
    # distance = (distance - distance.min()) / (distance.max() - distance.min())
    # distance = torch.tanh(distance)
    return torch.from_numpy(distance)

def space_similarity(space1, space2):
    return F.cosine_similarity(space1, space2).mean()

def space_kl_diverse(sapce1, space2):
    return F.kl_div(space1, space2)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='~/datasets/cifar10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10, shuffle=True, num_workers=4)

    # masks = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_all20_bn20/bnnas_c10_all20_bn20_search/2021-10-27_05-33-45/pareto_json/*.json')[:10]
    # ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_all20_bn20/*_nodepth_finetune*/*/checkpoints/*/*.ckpt')
    # # masks = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas/bnnas_random/*.json')[:10]
    # # ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_random_nodepth/*/*/checkpoints/*/*.ckpt')
    masks = glob('/home/xihe/xinhe/hyperbox/logs/runs/nasbench201_finetune/masks/nasbench201_mask_*.json')
    ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/nasbench201_finetune/nasbench201_finetune*/*/checkpoints/*/*.ckpt')
    ckpts = sorted(ckpts)
    masks = sorted(masks)
    networks = {}
    features = {}

    for batch_idx, (imgs, labels) in enumerate(testloader):
        if batch_idx==2:
            imgs = imgs.to(device)
            labels = labels.to(device)
            print(labels)
            batch_size = imgs.shape[0]
            break
    
    for idx, mask in enumerate(masks):
        # net = BNNet(num_classes=10, mask=mask, search_depth=False).to(device)
        net = NASBench201Network(mask=masks[idx]).to(device)
        load_state_dict(net, ckpts[idx])
        net.eval()
        networks[idx] = net
        y = net(imgs)
        print(idx, net.arch)
        features[idx] = networks[idx].features_list

    num_feats = len(features[0])
    for i in range(num_feats):
        dist_matrices = []
        for feat in features.values():
            dist = feat_distance(feat[i])
            dist_matrices.append(dist.view(batch_size, -1))
        dist_matrices = torch.stack(dist_matrices)
        titles = [f"Cell{i} Distance{j}" for j in range(len(dist_matrices))]
        titles=None
        show_images(dist_matrices.detach().cpu(), i, titles=titles, filename=f'cell{i}_dist.png')
        
        # distance = feat_distance(dist_matrices.view(len(features.values()), -1))
        # # print(f"the {i}-th matrices similarity: {distance}")
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # im = ax.imshow(distance.detach().cpu())
        # fig.savefig(f'similarity_cell{i}.png')
    
    print('done')        
