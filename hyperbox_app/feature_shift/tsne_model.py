'''
https://github.com/shivanichander/tSNE/blob/master/Code/tSNE%20Code.ipynb
'''

import itertools
import os
import random
from glob import glob

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

#Import scikitlearn for machine learning functionalities
import sklearn
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.datasets import load_digits  # For the UCI ML handwritten digits dataset
from sklearn.manifold import TSNE
from torchvision.utils import make_grid

from hyperbox.datamodules import CIFAR10DataModule
from hyperbox_app.feature_shift.bnnas import BNNet
from hyperbox_app.feature_shift.nasbench201 import NASBench201Network
from hyperbox_app.feature_shift.twonn import estimate_id


def load_state_dict(net, ckptfile):
    ckpt = torch.load(ckptfile, map_location='cpu')
    weights = {}
    for key, value in ckpt['state_dict'].items():
        if 'network' in key:
            weights[key.replace('network.', '')] = value
    net.load_state_dict(weights)


def make_gif(frame_folder, filename="my_awesome.gif"):
    frames = [Image.open(image) for image in glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save(filename, format="GIF", append_images=frames,
            save_all=True, duration=800, loop=0)

def plot(x, colors, index, num_classes=10, filename='tsne.png'):
  
    palette = np.array(sb.color_palette("hls", num_classes))  #Choosing color palette 

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    # Add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    plt.title(f"layer{index}")
    plt.savefig(filename)
    return f, ax, txts


device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='~/datasets/cifar10', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=True, num_workers=4)

for batch_idx, (imgs, labels) in enumerate(testloader):
    if batch_idx==2:
        imgs = imgs.to(device)
        labels = labels.to(device)
        print(labels)
        batch_size = imgs.shape[0]
        break


masks = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas/bnnas_random/*.json')[:10]
ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_random_nodepth/*/*/checkpoints/*/*.ckpt')
# masks = glob('/home/xihe/xinhe/hyperbox/logs/runs/nasbench201_finetune/masks/nasbench201_mask_*.json')
# ckpts = glob('/home/xihe/xinhe/hyperbox/logs/runs/nasbench201_finetune/nasbench201_finetune*/*/checkpoints/*/*.ckpt')
ckpts = sorted(ckpts)
masks = sorted(masks)

getAcc = lambda ckpt: ckpt.split('acc=')[1].split('.ckpt')[0]
accs = [float(getAcc(ckpt)) for ckpt in ckpts]
indices = np.argsort(accs)[::-1]
ckpts = np.array(ckpts)[indices]
masks = np.array(masks)[indices]
ckpts[:2], masks[:2]


networks = {}
features = {}
model_ids = {}
for idx, mask in enumerate(masks):
    net = BNNet(num_classes=10, mask=mask, search_depth=False).to(device)
    # net = NASBench201Network(mask=masks[idx]).to(device)
    load_state_dict(net, ckpts[idx])
    net.eval()
    networks[idx] = net
    y = net(imgs)
    features[idx] = networks[idx].features_list
    model_ids[idx] = {}


for i in range(10):
    crt_features = features[i]
    root = '/home/xihe/xinhe/hyperbox/hyperbox_app/feature_shift/tmp_tsne'
    os.system(f'rm {root}/*.png')
    for idx, feat in enumerate(crt_features):
        digits_final = TSNE(perplexity=100, learning_rate='auto').fit_transform(feat.detach().cpu().numpy())
        filename = os.path.join(root, f"nas201_{idx}.png")
        plot(digits_final,labels.cpu().numpy(),idx,10,filename)

    filename = os.path.join(root, f'tsne_bnnas_{i}_bs512.gif')
    make_gif(root, filename)
