""" Exponential Moving Average (EMA) of model updates
Hacked together by / Copyright 2020 Ross Wightman

source code: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
"""
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

from hyperbox.networks.base_nas_network import BaseNASNetwork


class ModelEma(nn.Module):
    """ Model Exponential Moving Average V2
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.7, final_decay=0.999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        if isinstance(model, BaseNASNetwork):
            self.module = model.copy()
        else:
            self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.init_decay = decay
        self.final_decay = final_decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def update_decay(self, epoch, all_epoch=100):
        self.decay = self.init_decay + (epoch/all_epoch)*(self.final_decay-self.init_decay)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from hyperbox.networks.ofa import OFAMobileNetV3
    from hyperbox.mutator import RandomMutator

    supernet = OFAMobileNetV3(num_classes=10, width_mult=1.0).to(device).eval()
    mutator = RandomMutator(supernet)
    # ema = ModelEma(supernet, 0.9)
    ema = ModelEma(supernet, 0.9).eval()
    print('init\nsupernet', supernet.classifier.weight.data[:2,:8])
    print('ema', ema.module.classifier.weight.data[:2,:8])
    for i in range(10):
        mutator.reset()
        supernet.init_weights()
        ema.update(supernet)
        x = torch.rand(2,3,64,64).to(device)
        y1 = supernet(x)
        subnet = ema.module.build_subnet(mutator._cache).cuda().eval()
        y2 = subnet(x)
        print('supernet', supernet.classifier.weight.data[:2,:8])
        print('ema', ema.module.classifier.weight.data[:2,:8])
