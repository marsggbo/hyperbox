import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as M

__all__ = [
    'vgg_forward',
    'resent_forward',
    'mbv3_forward',
    'mnas_forward',
    'densenet_forward',
    'squeeze_forward'
]

def vgg_forward(self, x):
    bs = x.shape[0]
    self.features_list = []
    for module in self.features:
        x = module(x)
        self.features_list.append(x.view(bs, -1))
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    self.features_list.append(x.view(bs, -1))
    x = self.classifier(x)
    return x

def resent_forward(self, x):
    bs = x.shape[0]
    self.features_list = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    self.features_list.append(x.view(bs, -1))

    x = self.layer1(x)
    self.features_list.append(x.view(bs, -1))
    x = self.layer2(x)
    self.features_list.append(x.view(bs, -1))
    x = self.layer3(x)
    self.features_list.append(x.view(bs, -1))
    x = self.layer4(x)
    self.features_list.append(x.view(bs, -1))

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    self.features_list.append(x.view(bs, -1))
    x = self.fc(x)

    return x

def mbv3_forward(self, x):
    bs = x.shape[0]
    self.features_list = []
    for module in self.features:
        x = module(x)
        self.features_list.append(x.view(bs, -1))

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    self.features_list.append(x.view(bs, -1))
    x = self.classifier(x)

    return x

def mnas_forward(self, x):
    bs = x.shape[0]
    self.features_list = []
    for module in self.layers:
        x = module(x)
        self.features_list.append(x.view(bs, -1))
    # Equivalent to global avgpool and removing H and W dimensions.
    x = x.mean([2, 3])
    return self.classifier(x)

def densenet_forward(self, x):
    bs = x.shape[0]
    self.features_list = []
    for module in self.features:
        x = module(x)
        self.features_list.append(x.view(bs, -1))

    out = F.relu(x, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    self.features_list.append(x.view(bs, -1))
    out = self.classifier(out)
    return out

def squeeze_forward(self, x):
    bs = x.shape[0]
    self.features_list = []
    for module in self.features:
        x = module(x)
        self.features_list.append(x.view(bs, -1))
    x = self.classifier(x)
    return torch.flatten(x, 1)
