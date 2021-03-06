import functools

from peewee import fn
from playhouse.shortcuts import model_to_dict
from hyperbox.networks.nasbench101.db_gen.model import Nb101TrialStats, Nb101TrialConfig
from hyperbox.networks.nasbench101.db_gen.graph_util import hash_module, infer_num_vertices


def query_nb101_trial_stats(arch, num_epochs, isomorphism=True, reduction=None):
    """
    Query trial stats of NAS-Bench-101 given conditions.

    Parameters
    ----------
    arch : dict or None
        If a dict, it is in the format that is described in
        :class:`nni.nas.benchmark.nasbench101.Nb101TrialConfig`. Only trial stats
        matched will be returned. If none, architecture will be a wildcard.
    num_epochs : int or None
        If int, matching results will be returned. Otherwise a wildcard.
    isomorphism : boolean
        Whether to match essentially-same architecture, i.e., architecture with the
        same graph-invariant hash value.
    reduction : str or None
        If 'none' or None, all trial stats will be returned directly.
        If 'mean', fields in trial stats will be averaged given the same trial config.

    Returns
    -------
    generator of dict
        A generator of :class:`nni.nas.benchmark.nasbench101.Nb101TrialStats` objects,
        where each of them has been converted into a dict.
    """
    fields = []
    if reduction == 'none':
        reduction = None
    if reduction == 'mean':
        for field_name in Nb101TrialStats._meta.sorted_field_names:
            if field_name not in ['id', 'config']:
                fields.append(fn.AVG(getattr(Nb101TrialStats, field_name)).alias(field_name))
    elif reduction is None:
        fields.append(Nb101TrialStats)
    else:
        raise ValueError('Unsupported reduction: \'%s\'' % reduction)
    query = Nb101TrialStats.select(*fields, Nb101TrialConfig).join(Nb101TrialConfig)
    conditions = []
    if arch is not None:
        if isomorphism:
            num_vertices = infer_num_vertices(arch)
            conditions.append(Nb101TrialConfig.hash == hash_module(arch, num_vertices))
        else:
            conditions.append(Nb101TrialConfig.arch == arch)
    if num_epochs is not None:
        conditions.append(Nb101TrialConfig.num_epochs == num_epochs)
    if conditions:
        query = query.where(functools.reduce(lambda a, b: a & b, conditions))
    if reduction is not None:
        query = query.group_by(Nb101TrialStats.config)
    for k in query:
        yield model_to_dict(k)
