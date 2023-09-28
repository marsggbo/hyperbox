import random

from hyperbox.networks.nasbenchasr.utils import recursive_iter, flatten, copy_structure


all_ops = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']
ops_no_zero = all_ops[:-1]
default_nodes = 3


def get_search_space(ops=None, nodes=None):
    ''' Return boundaries of the search space for the given list
        of available operations and number of nodes. 
    '''
    ops = ops if ops is not None else all_ops
    nodes = nodes if nodes is not None else default_nodes
    search_space = [[len(ops)] + [2]*(idx+1) for idx in range(nodes)]
    return search_space


def get_model_hash(arch_vec, ops=None, minimize=True):
    ''' Get hash of the architecture specified by arch_vec.
        Architecture hash can be used to determine if two
        configurations from the search space are in fact the
        same (graph isomorphism).
    '''
    from .graph_utils import get_model_graph, graph_hash
    g, _ = get_model_graph(arch_vec, ops=ops, minimize=minimize)
    return graph_hash(g)


def get_all_architectures(ops=None, nodes=None):
    ''' Yields all architecture configurations in the search space
    '''
    search_space = get_search_space(ops, nodes)
    flat = flatten(search_space)
    cfg = [0 for _ in range(len(flat))]
    end = False
    while not end:
        yield copy_structure(cfg, search_space)
        for dim in range(len(flat)):
            cfg[dim] += 1
            if cfg[dim] != flat[dim]:
                break
            cfg[dim] = 0
            if dim+1 >= len(flat):
                end = True


def get_random_architectures(num, ops=None, nodes=None, seed=None):
    ''' Get random architecture configurations from the search space
    '''
    ops = ops if ops is not None else all_ops
    nodes = nodes if nodes is not None else default_nodes
    if seed is not None:
        random.seed(seed)
    search_space = [[len(ops)] + [2]*(idx+1) for idx in range(nodes)]
    flat = flatten(search_space)
    models = []
    while len(models) < num:
        m = [random.randrange(opts) for opts in flat]
        m = copy_structure(m, search_space)
        models.append(m)
    return models


def get_archs_with_zero():
    models_with_zero = {}
    for m in get_all_architectures(all_ops, default_nodes):
        if 5 in flatten(m):
            h = get_model_hash(m)
            models_with_zero[h] = m
    new_model_archs = [models_with_zero[k] for k in sorted(models_with_zero.keys())]
    return new_model_archs


def arch_vec_to_names(arch_vec, ops=None):
    ''' Translates identifiers of operations in ``arch_vec`` to their names.
        ``ops`` can be provided externally to avoid relying on the current definition
        of available ops. Otherwise canonical ``all_ops`` will be used.
    '''

    if ops is None:
        ops = all_ops

    # current approach is to have an arch vector contain sub-vectors for node in a cell,
    # each subvector has a form of:
    # [op_idx, branch_op_idx...]
    # where op_idx points to an operation from ``all_ops`` and ``branch_op_idx`` is
    # either 0 (no skip connection) or 1 (identity skip connection)
    # since skip connects are already quite self-explanatory we leave them as they are
    # and only change numbers of the main operations to their respective names
    return [[all_ops[op_idx]] + branches for op_idx, *branches in arch_vec]
