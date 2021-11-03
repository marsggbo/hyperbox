
import numpy as np

import torch
import torch.nn as nn

from hyperbox.mutables.ops import Conv2d, Linear, BatchNorm2d
from hyperbox.mutables.spaces import ValueSpace, OperationSpace
from hyperbox.utils.utils import load_json, hparams_wrapper
from hyperbox.networks.base_nas_network import BaseNASNetwork

from hyperbox.networks.bnnas.bn_blocks import blocks_dict, InvertedResidual, conv_bn, conv_1x1_bn


@hparams_wrapper
class BNNet(BaseNASNetwork):
    def __init__(
        self,
        first_stride: int=1,
        first_channels: int=24,
        width_mult: int=1,
        channels_list: list=[32, 40, 80, 96, 192, 320],
        num_blocks: list=[2, 2, 4, 4, 4, 1],
        strides_list: list=[2, 2, 2, 1, 2, 1],
        num_classes: int=1000,
        search_depth: bool=True,
        is_only_train_bn: bool=True,
        mask: dict=None
    ):
        super(BNNet, self).__init__(mask)
        self.num_layers = len(channels_list)
        channels = int(first_channels * width_mult)
        self.first_conv = nn.Sequential(
            conv_bn(3, 40, first_stride),
            InvertedResidual(40, channels, 3, 1, 1, 1)
        )

        self.block_group_info = []
        self.features = nn.ModuleDict()
        c_in = channels
        for layer_id in range(self.num_layers):
            c_out = int(channels_list[layer_id] * width_mult)
            stride = strides_list[layer_id]
            n_blocks = num_blocks[layer_id]
            self.block_group_info.append([i for i in range(n_blocks)])

            ops = nn.Sequential()
            for i in range(n_blocks):
                key=f"layer{layer_id}_{i}"
                if i != 0:
                    stride = 1
                op = OperationSpace(
                    [block(c_in, c_out, stride) for block in blocks_dict.values()],
                    mask=self.mask, key=key
                )
                c_in = c_out
                ops.add_module(key, op)
            self.features[f"layer{layer_id}"] = ops

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(c_out, num_classes)
        )

        # dynamic depth
        self.runtime_depth = []
        for idx, block_group in enumerate(self.block_group_info):
            self.runtime_depth.append(
                ValueSpace(
                    list(range(1, len(block_group)+1)), key=f"depth{idx+1}", mask=self.mask)
            )
        self.runtime_depth = nn.Sequential(*self.runtime_depth)

        if is_only_train_bn:
            print(f"Only train BN.")
            self.freeze_except_bn()

    def forward(self, x):
        bs = x.shape[0]
        x = self.first_conv(x)

        if self.search_depth:
            for stage_id, key in enumerate(self.features.keys()):
                block_group = self.block_group_info[stage_id]
                depth = self.runtime_depth[stage_id].value
                active_idx = block_group[:depth]
                for idx in active_idx:
                    x = self.features[key][idx](x)
        else:
            for key in self.features:
                x = self.features[key](x)
        
        x = self.avgpool(x)
        x = x.view(bs, -1)
        x = self.classifier(x)
        return x

    def bn_metrics(self):
        values = 0
        for name, module in self.named_modules():
            if isinstance(module, OperationSpace):
                index = np.argmax(module.mask.numpy())
                op = module.candidates[index]
                value = op.conv[-1].weight.detach().mean()
                values += value
        return values

    def freeze_except_bn(self):
        for name, params in self.named_parameters():
            params.requires_grad = False

        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.requires_grad_(True)

    def defrost_all_params(self):
        for name, params in self.named_parameters():
            params.requires_grad = True

    def freeze_all_params(self):
        for name, params in self.named_parameters():
            params.requires_grad = False

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from hyperbox.mutator import RandomMutator
    from pytorch_lightning.utilities.seed import seed_everything
    net = BNNet(search_depth=False, is_only_train_bn=False, num_classes=10,
        channels_list=[32],
        num_blocks=[2],
        strides_list=[2],
    ).to(device)
    rm = RandomMutator(net)

    num = 10
    for i in range(5):
        seed_everything(i)
        rm.reset()
        x = torch.randn(num,3,64,64).to(device)
        preds = net(x)
        print(preds.argmax(-1))
    net = net.eval()
    for i in range(5):
        # rm.reset()
        x = torch.randn(num,3,64,64).to(device)
        preds = net(x)
        print(preds.argmax(-1))

    # opt = torch.optim.SGD(net.parameters(), lr=0.01)
    # H = torch.nn.CrossEntropyLoss()
    # print(f"Supernet size: {net.arch_size((2,3,64,64), 1, 1)}")
    # rm = RandomMutator(net)
    # for i in range(30):
    #     x = torch.rand(64,3,64,64).to(device)
    #     y = torch.randint(0,10,(64,)).to(device)
    #     opt.zero_grad()
    #     # rm.reset()
    #     print(f"Subnet size: {net.arch_size((2,3,64,64), 1, 1)}")
    #     print(net.bn_metrics())
    #     pred = net(x)
    #     loss = H(pred,y)
    #     loss.backward()
    #     conv = net.features['layer0'][0].candidates[-1].conv[0]
    #     bn = net.features['layer0'][0].candidates[-1].conv[7]
    #     linear = net.classifier[0]
    #     print('conv', conv.weight[0,:5,...].view(-1).detach(), conv.weight.requires_grad)
    #     print('bn', bn.weight[:5], bn.weight.requires_grad)
    #     print('linear', linear.weight[:5,0].view(-1).detach(), linear.weight.requires_grad)
    #     print('loss', loss.item())
    #     opt.step()
    #     if 6>i>3:
    #         net.freeze_except_bn()
    #     elif i > 6:
    #         net.defrost_all_params()
    #     pass
