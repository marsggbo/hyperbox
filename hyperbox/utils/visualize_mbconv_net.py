import graphviz


class ColorSet:
    hex_colors1 = [
        '#2878b5',
        '#9ac9db',
        '#f8ac8c',
        '#c82423',
        '#ff8884',
    ]
    hex_colors2 = [
        '#BEB8DC',
        '#E7DAD2',
        '#8ECFC9',
        '#FFBE7A',
        '#FA7F6F',
        '#82B0D2',
    ]

    def __init__(self, color_set='hex_colors1'):
        if color_set == 'hex_colors1':
            self._colors = self.hex_colors1
        elif color_set == 'hex_colors2':
            self._colors = self.hex_colors2
        else:
            raise ValueError('color_set must be one of hex_colors1 or hex_colors2')

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = value


def draw_arch(arch, index2op, filename='arch', color_set='hex_colors1'):
    colorset = ColorSet(color_set)
    index2color = {
        i: colorset.colors[i] for i in range(len(set(arch)))
    }

    dot = graphviz.Graph(comment='The Round Table')
    # dot.graph_attr['rankdir'] = 'LR'
    dot.graph_attr['rotate'] = '90'
    prev_index = None
    for idx, op_idx in enumerate(arch):
        op_name = index2op[op_idx]
        op_color = index2color[op_idx]
        op_index = f"{idx}"
        dot.node(op_index, label=op_name, style='filled', fillcolor=op_color, shape='box', fontsize='10')
        if idx == 0:
            prev_index = op_index
        else:
            dot.edge(op_index, prev_index)
            prev_index = op_index
    dot.render(directory='visual_output', view=False, filename=filename)

if __name__ == '__main__':
    arch = [
    0,1,2,1,2,3,1,1,2,2
    ]
    index2op = {
        0: 'MBConv3',
        1: 'MBConv5',
        2: 'MBConv7',
        3: 'Identity'
    }
    draw_arch(arch, index2op)
