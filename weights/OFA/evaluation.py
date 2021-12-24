import numpy as np
import torch
from ofa.imagenet_classification.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3 as OFA1
from tqdm import tqdm

from hyperbox.datamodules.imagenet_datamodule import ImagenetDataModule
from hyperbox.mutator import RandomMutator
from hyperbox.networks.ofa.ofa_mbv3 import OFAMobileNetV3 as OFA2
from hyperbox.networks.utils import set_running_statistics
from hyperbox.utils.metrics import accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(subnet, loader):
    iter_ = tqdm(loader)
    with torch.no_grad():
        top1 = []
        top5 = []
        subnet.eval()
        for i, (images, labels) in enumerate(iter_):
            images = images.to(device)
            labels = labels.to(device)
            output = subnet(images)
            acc = accuracy(output, labels, (1,5))
            top1.append(acc[0])
            top5.append(acc[1])
            crt_top1 = torch.tensor(top1).mean().item()
            crt_top5 = torch.tensor(top5).mean().item()
            iter_.set_description(f"Acc top-1: {crt_top1:.4f}, top-5: {crt_top5:.4f}")
        print('final', torch.tensor(top1).mean(), torch.tensor(top5).mean())


if __name__ == '__main__':
    # net1 = OFA1(dropout_rate=0, width_mult=1.0, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4])
    # net1.load_state_dict(
    #     torch.load('/home/xihe/xinhe/once-for-all/.torch/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0')['state_dict']
    # )
    # net1.set_active_subnet(ks=7, e=6, d=4)
    # subnet1 = net1.get_active_subnet(preserve_weight=True)
    
    net2 = OFA2(first_stride=2)
    net2.load_state_dict(
        torch.load('/home/xihe/xinhe/hyperbox/weights/OFA/OFA_MBV3_k357_d234_e46_w1.pth')
    )
    print('prepare supernet')

    imagenet = ImagenetDataModule('/datasets/imagenet2012/raw/', classes=1000)
    imagenet.setup('test')
    print('prepare dataset')
    
    rm = RandomMutator(net2)
    for i in range(2):
        if i==0:
            mask = net2.gen_mask(depth=4, expand_ratio=6, kernel_size=7)
        else:
            rm.reset()
            mask = rm._cache
        subnet = net2.build_subnet(mask).to(device)
        print('prepare subnet')
        set_running_statistics(subnet, imagenet.test_dataloader())
        print('set_running_statistics')
        evaluate(subnet, imagenet.test_dataloader())
        print('evaluate')
