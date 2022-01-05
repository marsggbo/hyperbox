import os
import copy

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
        inplanes,
        midplanes,
        outplanes,
        stride=1,
        downsample=None,
        groups=1,
        dilation=1,
    ):
        super(BasicBlock, self).__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Sequential(
            Conv2d(inplanes, midplanes, 3, stride, padding=1),
            BatchNorm2d(midplanes),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            Conv2d(midplanes, outplanes, 3, 1, padding=1),
            BatchNorm2d(outplanes),
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
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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
        block,
        layers,
        num_classes=100,
        zero_init_residual=False,
        groups=1,
        replace_stride_with_dilation=None,
        mask=None, # bool mask for ValueSpace
    ):
        super(ResNet, self).__init__()
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

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.inplanes = ValueSpace([4, 8, 12, 16], mask=self.mask)
        self.conv0 = nn.Sequential(
            Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, [4, 8, 12, 16], layers[0])
        self.layer2 = self._make_layer(
            block,  [4, 8, 12, 16, 20, 24, 28, 32], layers[1], stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block,  [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,60, 64],
            layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            Linear(self.inplanes, num_classes)
        )

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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        mid_planes = ValueSpace(planes, mask=self.mask)
        v_planes1 = ValueSpace(planes, mask=self.mask)
        # if stride != 1 or self.inplanes.value != v_planes1.value * block.expansion:
            # downsample = Conv2d(self.inplanes, v_planes1, 1, stride, act_func=None)
        downsample = nn.Sequential(
            Conv2d(self.inplanes, v_planes1, 1, stride),
            BatchNorm2d(v_planes1),
            # nn.ReLU()
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                mid_planes,
                v_planes1,
                stride,
                downsample,
                self.groups,
                previous_dilation,
            )
        )
        inplanes = v_planes1
        for _ in range(1, blocks):
            mid_planes = ValueSpace(planes, mask=self.mask)
            v_planes2 = ValueSpace(planes, mask=self.mask)
            downsample = nn.Sequential(
                Conv2d(inplanes, v_planes2, 1, 1),
                BatchNorm2d(v_planes2),
                # nn.ReLU()
                )
            layers.append(
                block(
                    inplanes,
                    mid_planes,
                    v_planes2,
                    downsample=downsample,
                    groups=self.groups,
                    dilation=self.dilation,
                )
            )
            inplanes = v_planes2
        self.inplanes = v_planes2

        return nn.Sequential(*layers)

    @property
    def arch(self):
        # _mutables = self.inplanes.mutator.mutables
        # vc = sorted(list(iter(_mutables)), key=lambda x:int(x.key.split('ValueSpace')[1]))
        vc = [m for m in self.modules() if isinstance(m, ValueSpace)]
        vc = sorted(vc, key=lambda x:int(x.key.split('ValueSpace')[1]))
        self._arch = '-'.join([str(x.value) for x in vc])
        return self._arch

    def forward(self, x):
        x = self.conv0(x)
        # x = self.maxpool(x) # remove for cifar100

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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


def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
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
