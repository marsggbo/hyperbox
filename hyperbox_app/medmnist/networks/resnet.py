import torch
import torch.nn as nn
import torchvision.models as M

from acsconv.converters import ACSConverter, Conv3dConverter, Conv2_5dConverter


def ResNet18_3D(
    in_channels=3,
    num_classes=10,
    pretrained=False,
    i3d_repeat_axis=None
):
    net = M.resnet18(pretrained=pretrained)

    if in_channels != 3 or min(net.conv1.kernel_size)!=3:
        net.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
    if num_classes != 1000:
        net.fc = nn.Linear(512, num_classes, bias=True)
    net_convert = Conv3dConverter(net, i3d_repeat_axis)
    return net_convert.model


def ResNet50_3D(
    in_channels=3,
    num_classes=10,
    pretrained=False,
    i3d_repeat_axis=None
):
    net = M.resnet50(pretrained=pretrained)

    if in_channels != 3:
        net.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
    if num_classes != 1000:
        net.fc = nn.Linear(512, num_classes, bias=True)
    net_convert = Conv3dConverter(net, i3d_repeat_axis)
    return net_convert.model


if __name__ == '__main__':
    net = ResNet18_3D()
    pass