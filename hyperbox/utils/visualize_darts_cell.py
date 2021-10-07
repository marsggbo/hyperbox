from collections import namedtuple
import sys
import os
import json
import numpy as np
from graphviz import Digraph
from argparse import ArgumentParser

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

def convert_genotypes(arch_json_file):
    with open(arch_json_file, 'r') as f:
        arch = json.load(f)
    normal = []
    reduce = []
    ops = []
    indices = []
    for key in arch:
        if 'switch' not in key:
            if sum(arch[key]) == 0:
                op = 'None'
            else:
                op = np.array(PRIMITIVES)[arch[key]][0]
            ops.append(op)
        else:
            _ops = []
            for i in range(len(arch[key])):
                if arch[key][i]:
                    indices.append(i)
                    _ops.append(ops[i])
            for op, idx in zip(_ops, indices):
                if 'norm' in key:
                    normal.append((op, idx))
                else:
                    reduce.append((op, idx))
            ops = []
            indices = []
    geno = Genotype(
        normal=normal, normal_concat=list(range(2, 2+len(normal)//2)),
        reduce=reduce, reduce_concat=list(range(2, 2+len(reduce)//2)),
    )
    print(geno)
    return geno

def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            if op != 'None':
                g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=True)


if __name__ == '__main__':
    parser = ArgumentParser('Visualize_DARTS')
    parser.add_argument('--file', type=str, help="the path of mask (json) file")
    args = parser.parse_args()

    arch_file = args.file
    assert os.path.exists(arch_file), f"{arch_file} not found."
    genotype = convert_genotypes(arch_file)
    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
