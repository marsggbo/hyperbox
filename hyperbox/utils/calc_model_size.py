import logging
from collections import OrderedDict

import torch
import torch.nn as nn

__all__ = ["flops_size_counter"]


def count_convNd(m, _, y):
    cin = m.in_channels
    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()
    total_ops = cin * output_elements * ops_per_element // m.groups  # cout x oW x oH
    m.total_ops = torch.Tensor([int(total_ops)])
    m.module_used = torch.tensor([1])


def count_linear(m, _, __):
    total_ops = m.in_features * m.out_features
    m.total_ops = torch.Tensor([int(total_ops)])
    m.module_used = torch.tensor([1])


def count_naive(m, _, __):
    m.module_used = torch.tensor([1])


register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.Linear: count_linear,
}


def flops_size_counter(_model, input_size):
    handler_collection = []
    logger = logging.getLogger(__name__)
    if isinstance(_model, nn.DataParallel):
        model = _model.module
    else:
        model = _model

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.register_buffer('total_ops', torch.zeros(1))
        m_.register_buffer('total_params', torch.zeros(1))
        m_.register_buffer('module_used', torch.zeros(1))

        for p in m_.parameters():
            m_.total_params += torch.Tensor([p.numel()])

        m_type = type(m_)
        fn = register_hooks.get(m_type, count_naive)

        if fn is not None:
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    def remove_buffer(m_):
        if len(list(m_.children())) > 0:
            return

        del m_.total_ops, m_.total_params, m_.module_used

    original_device = next(model.parameters()).device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    assert isinstance(input_size, tuple)
    if torch.is_tensor(input_size[0]):
        x = (t.to(original_device) for t in input_size)
    else:
        x = (torch.zeros(input_size).to(original_device), )
    with torch.no_grad():
        model(*x)

    total_ops = 0
    total_params = 0
    for name, m in model.named_modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        if not m.module_used:
            continue
        total_ops += m.total_ops
        total_params += m.total_params
        # print("%s: %.2f %.2f" % (name, m.total_ops.item(), m.total_params.item()))

    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()
    model.apply(remove_buffer)

    total_ops, total_params = total_ops/1e6, total_params*4/1024**2
    # print(f"{total_ops} MFLOPS, {total_params} MB")
    result = OrderedDict({
        'flops': total_ops,
        'size': total_params,
    })
    return result
