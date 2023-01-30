import os
import copy
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn

from hyperbox.mutables.ops import Conv2d, Linear, BatchNorm2d
from hyperbox.mutables.spaces import ValueSpace
from hyperbox.utils.utils import load_json, hparams_wrapper

from hyperbox.networks.base_nas_network import BaseNASNetwork

__all__ = [
    "ResNet",
    "resnet18",
    "resnet20",
    "resnet34",
    "resnet50"
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: Union[ValueSpace, int],
        outplanes: Union[ValueSpace, int],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(BasicBlock, self).__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if norm_layer is None:
            norm_layer = BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Sequential(
            Conv2d(inplanes, outplanes, 3, stride, padding=1),
            norm_layer(outplanes),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            Conv2d(outplanes, outplanes, 3, 1, padding=1),
            norm_layer(outplanes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if out.shape != x.shape and self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: Union[ValueSpace, int],
        planes: Union[ValueSpace, int],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if isinstance(planes, int):
            width = int(planes * (base_width / 64.0)) * groups
        else:
            width = [int(p * (base_width / 64.0)) * groups for p in planes]
            mask = None if planes.is_search else planes.mask
            width = ValueSpace(width, key=planes.key, mask=mask)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            Conv2d(inplanes, width, 1, 1),
            norm_layer(width),
            self.relu
        )
        self.conv2 = nn.Sequential(
            Conv2d(width, width, 3, stride, padding=1, groups=groups, dilation=dilation),
            norm_layer(width),
            self.relu
        )
        expand_outplanes = planes * self.expansion
        self.conv3 = nn.Sequential(
            Conv2d(width, expand_outplanes, 1, 1),
            norm_layer(expand_outplanes),
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@hparams_wrapper
class ResNet(BaseNASNetwork):
    counter_subnet = 1
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]], # BasicBlock or Bottleneck
        num_layers: list, # [2, 2, 2, 2] for resnet18
        num_classes: int=1000, # 10 for CIFAR10, 1000 for ImageNet
        zero_init_residual: bool=False, # True for ImageNet
        groups: int=1,
        width_per_group: int = 64,
        replace_stride_with_dilation=None,
        ratios: list=[0.1, 0.2, 0.5, 0.8, 1], # channel ratios
        mask: dict=None, # dict of mask
    ):
        super(ResNet, self).__init__(mask)
        self.mask = load_json(mask)

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.inplanes = ValueSpace([int(64*x) for x in ratios], mask=self.mask, key='layer1_inplanes')
        self.conv0 = nn.Sequential(
            Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, [int(64*x) for x in ratios], num_layers[0],
            stride=1, dilate=False, prefix='layer1')
        self.layer2 = self._make_layer(block, [int(128*x) for x in ratios], num_layers[1],
            stride=2, dilate=replace_stride_with_dilation[0], prefix='layer2')
        self.layer3 = self._make_layer(block, [int(256*x) for x in ratios], num_layers[2],
            stride=2, dilate=replace_stride_with_dilation[1], prefix='layer3')
        self.layer4 = self._make_layer(block, [int(512*x) for x in ratios], num_layers[3],
            stride=2, dilate=replace_stride_with_dilation[2], prefix='layer4')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias: nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: callable, # BasicBlock or Bottleneck
        planes: list, # #channel size
        blocks: int, # num of blocks
        stride: int=1,
        dilate: bool=False,
        prefix: str="", # prefix for layer name
    ):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            
        layers = []
        inplanes = self.inplanes
        expansion = block.expansion
        for idx in range(blocks):
            outplanes = ValueSpace(planes, mask=self.mask, key=prefix+f'_outplanes{idx}')
            block_outplanes = outplanes * expansion
            stride = stride if idx == 0 else 1
            downsample = nn.Sequential(
                Conv2d(inplanes, block_outplanes, 1, stride),
                BatchNorm2d(block_outplanes),
                )
            layers.append(
                block(inplanes, outplanes, stride, downsample, self.groups, self.base_width, self.dilation)
            )
            inplanes = block_outplanes
        self.inplanes = block_outplanes

        return nn.Sequential(*layers)

    @property
    def arch(self):
        # _mutables = self.inplanes.mutator.mutables
        # vc = sorted(list(iter(_mutables)), key=lambda x:int(x.key.split('ValueSpace')[1]))
        vc = [m for m in self.modules() if isinstance(m, ValueSpace)]
        vc = sorted(vc, key=lambda x: x.key)
        self._arch = '-'.join([str(x.value) for x in vc])
        return self._arch

    def forward(self, x):
        x = self.conv0(x)
        # x = self.maxpool(x) # remove for cifar100

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if 'total_ops' in key or \
                'total_params' in key or \
                'module_used' in key:
                continue
            else:
                model_dict[key] = state_dict[key]
        super(ResNet, self).load_state_dict(model_dict, **kwargs)

    def build_subnet(self, mask):
        hparams = self.hparams.copy()
        new_mask = {}
        len_mask = len(mask)
        for key in mask:
            _id = key.split('ValueSpace')[-1]
            new_id = int(_id) + len_mask * self.counter_subnet
            new_key = f"ValueSpace{new_id}"
            new_mask[new_key] = mask[key].clone().detach()
        hparams['mask'] = new_mask
        subnet = ResNet(**hparams)
        self.counter_subnet += 1
        return subnet

    def load_from_supernet(self, state_dict, **kwargs):
        def sub_filter_start_end(kernel_size, sub_kernel_size):
            center = kernel_size // 2
            dev = sub_kernel_size // 2
            start, end = center - dev, center + dev + 1
            assert end - start == sub_kernel_size
            return start, end
        model_dict = self.state_dict()
        for key in state_dict:
            if 'total_ops' in key or \
                'total_params' in key or \
                'module_used' in key or \
                'mask' in key:
                continue
            if model_dict[key].shape == state_dict[key].shape:
                model_dict[key] = state_dict[key]
            else:
                shape = model_dict[key].shape
                if len(shape) == 1:
                    model_dict[key].data = state_dict[key].data[:shape[0]]
                if len(shape) == 2:
                    _out, _in = shape
                    model_dict[key].data = state_dict[key].data[:_out, :_in]
                if len(shape) == 4:
                    _out, _in, k, k = shape
                    k_larger = state_dict[key].shape[-1]
                    start, end = sub_filter_start_end(k_larger, k)
                    model_dict[key].data = state_dict[key].data[:_out, :_in, start:end, start:end]
        super(ResNet, self).load_state_dict(model_dict, **kwargs, strict=False)


def _resnet(arch, block, num_layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, num_layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )

def resnet20(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-20 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet20", BasicBlock, [3, 3, 3], pretrained, progress, device, **kwargs
    )

def resnet34(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )

def resnet50(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )


if __name__ == '__main__':
    from hyperbox.mutator import RandomMutator, OnehotMutator, DartsMutator
    net = resnet34(ratios=[0.5, 1])
    # net = resnet50()
    rm = RandomMutator(net)
    # rm = DartsMutator(net) # ValueSpace-based operations are not compatible with DartsMutator
    # rm = OnehotMutator(net)
    x = torch.rand(2, 3, 32, 32)
    for i in range(10):        
        rm.reset()
        y = net(x)
        print(y.shape)
        arch = net.arch
        print(arch, len(rm._cache))
