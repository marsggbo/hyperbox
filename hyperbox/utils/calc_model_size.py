import logging
from collections import OrderedDict

import torch
import torch.nn as nn

import hyperbox
from hyperbox.mutables.ops import FinegrainedModule

__all__ = ["flops_size_counter"]


def count_convNd(m, _, y):
    cin = m.in_channels
    kernel_ops = m.weight.shape[2:].numel()
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


def count_FG_convNd(m, _, y):
    cin = m.value_spaces['in_channels'].value if m.search_in_channel else m.in_channels
    kernel_size = m.value_spaces['kernel_size'].value if m.search_kernel_size else m.kernel_size
    if isinstance(kernel_size, int):
        dim = int(m.conv_dim[0])
        kernel_ops = kernel_size**dim
    elif isinstance(kernel_size, (list, tuple)):
        kernel_ops = torch.prod(torch.Tensor(kernel_size))
    ops_per_element = kernel_ops
    groups = m.value_spaces['groups'].value if m.search_groups else m.groups
    output_elements = y.nelement()
    total_ops = cin * output_elements * ops_per_element // groups  # cout x oW x oH
    m.total_ops = torch.Tensor([int(total_ops)])
    m.module_used = torch.tensor([1])


def count_FG_linear(m, _, __):
    in_features = m.value_spaces['in_features'].value if m.search_in_features \
        else m.in_features
    out_features = m.value_spaces['out_features'].value if m.search_out_features \
        else m.out_features
    total_ops = in_features * out_features
    m.total_ops = torch.Tensor([int(total_ops)])
    m.module_used = torch.tensor([1])


register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.Linear: count_linear,
    hyperbox.mutables.ops.conv.Conv1d: count_FG_convNd,
    hyperbox.mutables.ops.conv.Conv2d: count_FG_convNd,
    hyperbox.mutables.ops.conv.Conv3d: count_FG_convNd,
    hyperbox.mutables.ops.linear.Linear: count_FG_linear,
}


def flops_size_counter(_model, input_size, convert=True, verbose=False):
    '''Calculate the flops and size of the given model
    Args:
        _model: nn.Module
        input_size: Tuple[int, ...]
        convert: bool
            if True, convert to MFLOPs and MB
        verbose: bool. If True, print the info of the model
    '''
    handler_collection = []
    logger = logging.getLogger(__name__)
    if isinstance(_model, nn.DataParallel):
        model = _model.module
    else:
        model = _model

    def add_hooks(m_):
        if isinstance(m_, FinegrainedModule):
            pass
        elif len(list(m_.children())) > 0:
            return

        m_.register_buffer('total_ops', torch.zeros(1))
        m_.register_buffer('total_params', torch.zeros(1))
        m_.register_buffer('module_used', torch.zeros(1))

        if isinstance(m_, FinegrainedModule):
            if hasattr(m_, 'params'):
                params = m_.params
            else:
                params = sum([p.numel() for p in m_.parameters()])
            m_.total_params += params
        else:
            for p in m_.parameters():
                m_.total_params += torch.Tensor([p.numel()])

        m_type = type(m_)
        fn = register_hooks.get(m_type, count_naive)

        if fn is not None:
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    def remove_buffer(m_):
        if isinstance(m, FinegrainedModule):
            pass
        elif len(list(m.children())) > 0:
            return
        if hasattr(m_, 'total_ops'):
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

    total_ops = torch.zeros(1)
    total_params = torch.zeros(1)
    for name, m in model.named_modules():
        if isinstance(m, FinegrainedModule):
            pass
        elif len(list(m.children())) > 0:
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

    if convert:
        total_ops, total_params = total_ops/1e6, total_params*4/1024**2
    if verbose:
        if convert:
            print(f"{total_ops} MFLOPS, {total_params} MB")
        else:
            print(f"{total_ops} FLOPS, {total_params} #Params")
    result = OrderedDict({
        'flops': total_ops,
        'size': total_params,
    })
    return result
