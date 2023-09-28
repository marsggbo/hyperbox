import copy
import random
import hashlib
import tempfile
import subprocess
import collections.abc

import tqdm
import numpy as np
import networkx as nx

from hyperbox.networks.nasbenchasr.utils import flatten, count, get_first_n

_use_np = True


def get_model_graph_np(arch_vec, ops=None, minimize=True, keep_dims=False):
    if ops is None:
        from . import search_space as ss
        ops = ss.all_ops
    num_nodes = len(arch_vec)
    mat = np.zeros((num_nodes+2, num_nodes+2))
    labels = ['input']
    prev_skips = []
    for nidx, node in enumerate(arch_vec):
        op = node[0]
        labels.append(ops[op])
        mat[nidx, nidx+1] = 1
        for i, sc in enumerate(prev_skips):
            if sc:
                mat[i, nidx+1] = 1
        prev_skips = node[1:]
    labels.append('output')
    mat[num_nodes, num_nodes+1] = 1
    for i, sc in enumerate(prev_skips):
        if sc:
            mat[i, num_nodes+1] = 1
    orig = None
    if minimize:
        orig = copy.copy(mat), copy.copy(labels)
        for n in range(len(mat)):
            if labels[n] == 'zero':
                for n2 in range(len(mat)):
                    if mat[n,n2]:
                        mat[n,n2] = 0
                    if mat[n2,n]:
                        mat[n2,n] = 0
        def bfs(src, mat, backward):
            visited = np.zeros(len(mat))
            q = [src]
            visited[src] = 1
            while q:
                n = q.pop()
                for n2 in range(len(mat)):
                    if visited[n2]:
                        continue
                    if (backward and mat[n2,n]) or (not backward and mat[n,n2]):
                        q.append(n2)
                        visited[n2] = 1
            return visited
        vfw = bfs(0, mat, False)
        vbw = bfs(len(mat)-1, mat, True)
        v = vfw + vbw
        dangling = (v < 2).nonzero()[0]
        if dangling.size:
            if keep_dims:
                mat[dangling, :] = 0
                mat[:, dangling] = 0
                for i in dangling:
                    labels[i] = None
            else:
                mat = np.delete(mat, dangling, axis=0)
                mat = np.delete(mat, dangling, axis=1)
                for i in sorted(dangling, reverse=True):
                    del labels[i]
    return (mat, labels), orig

def get_model_graph_nx(arch_vector, ops=None, minimize=True, keep_dims=False):
    ''' Get :class:`netwworkx.DiGraph` object from an arch vector.
        If ``minimize`` is ``True``, the graph will be minimized by removing
        "zero" operations and consequently any dangling nodes.
    '''
    if ops is None:
        from . import search_space as ss
        ops = ss.all_ops
    num_nodes = len(arch_vector)
    g = nx.DiGraph()
    g.add_node(0, label='input')
    prev_skips = []
    for nidx, node in enumerate(arch_vector):
        op = node[0]
        g.add_node(nidx+1, label=ops[op])
        g.add_edge(nidx, nidx+1)
        for i, sc in enumerate(prev_skips):
            if sc:
                g.add_edge(i, nidx+1)
        prev_skips = node[1:]
    g.add_node(num_nodes+1, label='output')
    g.add_edge(num_nodes, num_nodes+1)
    for i, sc in enumerate(prev_skips):
        if sc:
            g.add_edge(i, num_nodes+1)
    orig = None
    if minimize:
        orig = copy.deepcopy(g)
        for n in dict(g.nodes):
            if g.nodes[n]['label'] == 'zero':
                g.remove_node(n)
        for _i in range(2):
            if 0 in g.nodes:
                from_source = nx.descendants(g, 0)
            else:
                from_source = []
            for n in dict(g.nodes):
                keep = True
                desc = nx.descendants(g, n)
                if n != num_nodes+1:
                    if num_nodes+1 not in desc:
                        keep = False
                if n > 0:
                    if n not in from_source:
                        keep = False
                if not keep:
                    if not _i:
                        if keep_dims:
                            edges = list(g.in_edges(n)) + list(g.out_edges(n))
                            g.remove_edges_from(edges)
                            g.nodes[n]['label'] = None
                        else:
                            g.remove_node(n)
                    else:
                        print(_i, n, desc)
                        show_graph(g)
                        show_graph(orig)
                        assert False
    return g, orig

def get_model_graph(arch_vector, ops=None, minimize=True, keep_dims=False):
    if _use_np:
        return get_model_graph_np(arch_vector, ops, minimize, keep_dims)
    else:
        return get_model_graph_nx(arch_vector, ops, minimize, keep_dims)


def graph_hash_np(g):
    from . import search_space as ss
    m, l = g
    def hash_module(matrix, labelling):
        """Computes a graph-invariance MD5 hash of the matrix and label pair.
        Args:
            matrix: np.ndarray square upper-triangular adjacency matrix.
            labelling: list of int labels of length equal to both dimensions of
                matrix.
        Returns:
            MD5 hash of the matrix and labelling.
        """
        vertices = np.shape(matrix)[0]
        in_edges = np.sum(matrix, axis=0).tolist()
        out_edges = np.sum(matrix, axis=1).tolist()
        assert len(in_edges) == len(out_edges) == len(labelling), f'{labelling} {matrix}'
        hashes = list(zip(out_edges, in_edges, labelling))
        hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
        # Computing this up to the diameter is probably sufficient but since the
        # operation is fast, it is okay to repeat more times.
        for _ in range(vertices):
            new_hashes = []
            for v in range(vertices):
                in_neighbours = [hashes[w] for w in range(vertices) if matrix[w, v]]
                out_neighbours = [hashes[w] for w in range(vertices) if matrix[v, w]]
                new_hashes.append(hashlib.md5(
                        (''.join(sorted(in_neighbours)) + '|' +
                        ''.join(sorted(out_neighbours)) + '|' +
                        hashes[v]).encode('utf-8')).hexdigest())
            hashes = new_hashes
        fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()
        return fingerprint
    labels = []
    if l:
        labels = [-1] + [ss.all_ops.index(op) for op in l[1:-1]] + [-2]
    return hash_module(m, labels)

def graph_hash_nx(g):
    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(g, node_attr='label')

def graph_hash(g):
    if _use_np:
        return graph_hash_np(g)
    else:
        return graph_hash_nx(g)


_op_to_node_color = {
    'linear': 'tomato',
    'conv5': 'cadetblue1',
    'conv5d2': 'deepskyblue1',
    'conv7': 'olivedrab2',
    'conv7d2': 'seagreen4'
}

_op_to_label = {
    'linear': 'Linear',
    'conv5': 'Conv(5)',
    'conv5d2': 'Conv(5,d=2)',
    'conv7': 'Conv(7)',
    'conv7d2': 'Conv(7,d=2)',
    'input': 'Input',
    'output': 'Output',
    'zero': 'Zero'
}


def _make_nice(agraph):
    positions = {}
    agraph.node_attr['shape'] = 'rectangle'
    agraph.node_attr['style'] = 'rounded'
    agraph.graph_attr['splines'] = 'true'
    agraph.graph_attr['esep'] = 0.17
    #agraph.graph_attr['overlap'] = 'false'
    for node in agraph.nodes():
        op = node.attr['label']
        node.attr['label'] = _op_to_label.get(op, op)
        node.attr['width'] = 1.2
        node.attr['height'] = 0.3
        if op in _op_to_node_color:
            node.attr['fillcolor'] = _op_to_node_color[op]
            node.attr['style'] = 'filled,rounded'

        positions[2*int(node)] = node


    outputs = {}
    removed = set()
    for e in agraph.edges():
        if e in removed:
            continue
        if int(e[0]) + 1 != int(e[1]):
            e.attr['group'] = 'branches'
            e.attr['style'] = 'dashed'
            d = int(e[1])
            prev = str(d-1)
            if prev not in outputs:
                onode = outputs.setdefault(prev, f'o{prev}')
                agraph.add_node(onode, label='+', shape='circle', width=0.3, height=0.3, fixedsize=True, fontsize=16)
                positions[2*int(prev)+1] = onode
                for e2 in agraph.edges():
                    if e2[0] == prev and e2[1] == e[1]:
                        agraph.remove_edge(e2)
                        removed.add(e2)
                agraph.add_edge(prev, onode, group='main', arrowsize=0.5)
                agraph.add_edge(onode, e[1], group='main', arrowsize=0.5)
            else:
                onode = outputs[prev]

            agraph.add_edge(outputs.get(e[0], e[0]), onode, group='branches', style='dashed', arrowsize=0.5)
            agraph.remove_edge(e)
            removed.add(e)
        else:
            e.attr['group'] = 'main'
            e.attr['arrowsize'] = 0.5

    _pos = sorted(positions.keys())
    p = 0
    next_half = False
    is_next_sc = [_pos[i+1] % 2 != 0 for i in range(len(_pos)-1)] + [False]
    is_prev_sc = [False] + is_next_sc[:-1]
    for pos, nsc, psc in zip(_pos, is_next_sc, is_prev_sc):
        node = agraph.get_node(positions[pos])
        node.attr['pos'] = f'0,{p}!'
        if not nsc and not psc:
            p -= 0.47
        else:
            p -= 0.47


def show_graph(g, aid=None, show=True, out_dir=None):
    ''' Renders graph ``g`` using graphiviz.
        ``aid`` is an optional architecture id, if provided,
        the rendered graph will be stored under "{out_dir}/nb_graph.{aid}.png".
        (If ``out_dir`` is ``None``, it will default to ``graphs``).
        Otherwise, it will be saved in a temporary file.
        If ``show`` is ``True``, the rendered file will be opened with "xdg-open".
    '''
    if _use_np:
        a, l = g
        g = nx.from_numpy_array(a, create_using=nx.DiGraph)
        for idx, label in enumerate(l):
            g.nodes[idx]['label'] = label
    a = nx.nx_agraph.to_agraph(g)
    _make_nice(a)
    a.layout('dot', '-Kfdp')
    if aid is None:
        fname = tempfile.mktemp('.png', 'nb_graph.')
    else:
        dname = out_dir if out_dir is not None else "graphs"
        fname = f'{dname}/nb_graph.{aid}.png'
    a.draw(fname)
    if show:
        subprocess.run(['xdg-open', fname], check=True)


def show_model(arch_vec, aid=None, show=True, inc_full=True, out_dir=None):
    ''' Renders graphs constructed from arch vector (both minimal and full).
        Full graph is only rendered if different from minimal.
        ``aid`` is an architecture id which will be used when saving rendered graphs,
        if not provided it will be derived from ``arch_vec``.
    '''
    g, full = get_model_graph(arch_vec)
    if aid is None:
        aid = '_'.join(map(str, flatten(arch_vec)))
    show_graph(g, aid=aid, show=show, out_dir=out_dir)
    if full is not None:
        if graph_hash(g) != graph_hash(full):
            assert 5 in flatten(arch_vec)
            show_graph(full, aid=f'{aid}_full', show=show, out_dir=out_dir)
        else:
            assert 5 not in flatten(arch_vec)


def compare_nx_and_np():
    from .search_space import get_all_architectures, all_ops, default_nodes
    global _use_np
    all_count = count(get_all_architectures(all_ops, default_nodes))
    _use_np = False
    all_hashes = set()
    without_zero = set()
    unique_graphs = []
    conflicts = {}
    for m in tqdm.tqdm(get_all_architectures(all_ops, default_nodes), total=all_count):
        has_zero = 5 in flatten(m)
        g, _ = get_model_graph(m)
        h = graph_hash(g)
        if h not in all_hashes:
            unique_graphs.append(m)
        else:
            conflicts[h] = m
        all_hashes.add(h)
        if not has_zero:
            without_zero.add(h)
    _use_np = True
    np_hashes = set()
    invalid = []
    for m in tqdm.tqdm(get_all_architectures(all_ops, default_nodes), total=all_count):
        has_zero = 5 in flatten(m)
        g, _ = get_model_graph(m)
        h = graph_hash(g)
        if h not in np_hashes:
            if m not in unique_graphs:
                invalid.append(m)
        np_hashes.add(h)
    print('Core:', len(without_zero))
    print('With zeros:', len(all_hashes))
    print('Unique:', len(unique_graphs))
    print('Np unique:', len(np_hashes))
    print('Invalid:', len(invalid))
    _use_np = False
    if invalid:
        inv = invalid[0]
        g, _ = get_model_graph(inv)
        h = graph_hash(g)
        conflicting = conflicts[h]
        show_model(invalid[0])
        show_model(conflicting)


def main():
    from .search_space import get_all_architectures, all_ops, default_nodes
    all_count = count(get_all_architectures(all_ops, default_nodes))
    all_hashes = set()
    without_zero = set()
    for m in tqdm.tqdm(get_all_architectures(all_ops, default_nodes), total=all_count):
        has_zero = 5 in flatten(m)
        g, _ = get_model_graph(m)
        h = graph_hash(g)
        all_hashes.add(h)
        if not has_zero:
            without_zero.add(h)
    # show_model([[0,1], [5,1,0], [3,1,1,1]])
    print('Core:', len(without_zero))
    print('With zeros:', len(all_hashes))


if __name__ == '__main__':
    main()
