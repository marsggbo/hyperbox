'''
Reference: https://github.com/SamsungLabs/nb-asr/blob/HEAD/nasbench_asr/model/torch/model.py
'''

import os

import torch
import torch.nn as nn

from hyperbox.mutables import spaces
from hyperbox.networks.base_nas_network import BaseNASNetwork

from hyperbox.networks.nasbenchasr.ops import PadConvRelu, _ops, _branch_ops
from hyperbox.networks.nasbenchasr.dataset import from_folder


class Node(nn.Module):
    def __init__(
        self,
        filters,
        node_index: int,
        prefix: str="",
        dropout_rate=0.0,
        mask=None
    ):
        super(Node, self).__init__()
        # main edge
        op_list = [
            op_cand(filters, filters, dropout_rate=dropout_rate) for op_name, op_cand in _ops.items()
        ]
        self.op = spaces.OperationSpace(op_list, mask=mask, key=f"{prefix}_node{node_index}_main")

        # skip branch edges
        num_branch_ops = node_index + 1
        branch_op_list = [
            op_cand() for op_cand in _branch_ops.values()
        ]
        self.branch_ops = []
        for s_idx in range(num_branch_ops):
            self.branch_ops.append(
                spaces.OperationSpace(branch_op_list, mask=mask, key=f"{prefix}_node{node_index}_skip{s_idx}")
            )
        self.branch_ops = nn.Sequential(*self.branch_ops)

    def forward(self, input_list):
        assert len(input_list) == len(self.branch_ops), 'Branch op and input list have different lenghts'

        output = self.op(input_list[-1])
        edges = [output] 
        for i in range(len(self.branch_ops)):
            x = self.branch_ops[i](input_list[i])
            edges.append(x)

        return sum(edges)


class SearchCell(nn.Module): 
    def __init__(
        self,
        filters: int,
        num_nodes: int=3,
        dropout_rate: float=0.0,
        use_norm: bool=True,
        prefix: str="",
        mask: dict=None
    ):
        super(SearchCell, self).__init__()
        self.mask = mask
        self.num_nodes = num_nodes

        self.nodes = nn.ModuleList()
        for node_index in range(self.num_nodes):
            node = Node(filters=filters, node_index=node_index, prefix=prefix, dropout_rate=dropout_rate, mask=mask)
            self.nodes.append(node)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm_layer = nn.LayerNorm(filters, eps=0.001)

    def forward(self, input):
        outputs = [input] # input is the output coming from node 0
        for node in self.nodes:
            n_out = node(outputs)
            outputs.append(n_out)
        output = outputs[-1] #last node is the output
        if self.use_norm:
            output = output.permute(0,2,1)
            output = self.norm_layer(output)
            output = output.permute(0,2,1)
        return output 


class NASBenchASR(BaseNASNetwork):

    NASBenchASR_DATAPATH = os.path.expanduser("~/.hyperbox/nasbenchasr")
    main_op_name2idx = {
        'linear': 0,
        'conv5': 1,
        'conv5d2': 2,
        'conv7': 3,
        'conv7d2': 4,
        'zero': 5,
    }
    main_op_idx2name = {  
        0: 'linear',  
        1: 'conv5',  
        2: 'conv5d2',  
        3: 'conv7',  
        4: 'conv7d2',  
        5: 'zero',  
    }  

    def __init__(
        self,
        num_blocks: int = 4,
        features: int = 80,
        filters: list = [600, 800, 1000, 1200], # the number of filters per block
        cnn_time_reduction_kernels: list = [8, 8, 8, 8],
        cnn_time_reduction_strides: list = [1, 1, 2, 2] ,
        scells_per_block: list = [3, 4, 5, 6], # the number of cells per block
        num_nodes: int=3, # number of nodes per cell
        num_classes: int=48,
        use_rnn: bool=True,
        use_norm: bool=True,
        dropout_rate: float=0.0,
        mask: dict=None
    ):
        '''
        Args:
        - mask: dict, e.g.,
            {
                'block0_cell0_node0_main': [1, 0, 0, 0, 0],
                'block0_cell0_node0_skip0': [1, 0],
                'block0_cell0_node1_main': [0, 1, 0, 0, 0],
                'block0_cell0_node1_skip0': [1, 0],
                'block0_cell0_node2_main': [0, 1, 0, 0, 0],
                'block0_cell0_node2_skip0': [1, 0],
                'block0_cell0_node2_skip1': [1, 0],
                'block0_cell0_node2_skip2': [0, 1],
                ...
                }
            }
        '''
        super(NASBenchASR, self).__init__(mask=mask)

        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.use_rnn = use_rnn
        self.use_norm = use_norm
        self.dropout_rate = dropout_rate

        self.num_blocks = num_blocks
        self.features = features
        self.filters = filters
        self.cnn_time_reduction_kernels = cnn_time_reduction_kernels
        self.cnn_time_reduction_strides = cnn_time_reduction_strides
        self.scells_per_block = scells_per_block
        
        layers = nn.ModuleList()

        for i in range(self.num_blocks):
            layers.append(PadConvRelu(
                in_channels= self.features if i==0 else self.filters[i-1], 
                out_channels=self.filters[i], 
                kernel_size=self.cnn_time_reduction_kernels[i], 
                dilation=1,
                strides=self.cnn_time_reduction_strides[i],
                groups=1,
                name=f'conv_{i}'))

            # TODO: normalize axis=1
            layers.append(nn.LayerNorm(self.filters[i], eps=0.001))

            for j in range(self.scells_per_block[i]):
                prefix = f"block{i}_cell{j}"
                cell = SearchCell(
                    filters=self.filters[i], num_nodes=self.num_nodes, use_norm=use_norm,
                    dropout_rate=dropout_rate, prefix=prefix, mask=mask) 
                layers.append(cell)

        if use_rnn:
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.LSTM(input_size=self.filters[self.num_blocks-1], hidden_size=500, batch_first=True, dropout=0.0))
            layers.append(nn.Linear(in_features=500, out_features=num_classes+1))
        else:
            layers.append(nn.Linear(in_features=self.filters[self.num_blocks-1], out_features=num_classes+1))

        # self._model = nn.Sequential(*layers)
        self.model = layers

    def forward(self, input): # input is (B, F, T)
        for xx in self.model:
            if isinstance(xx, nn.LSTM):
                input = input.permute(0,2,1)
                input = xx(input)[0]
                input = input.permute(0,2,1)
            elif isinstance(xx, nn.Linear):
                input = input.permute(0,2,1)
                input = xx(input)
            elif isinstance(xx, nn.LayerNorm):
                input = input.permute(0,2,1)
                input = xx(input)
                input = input.permute(0,2,1)
            else:
                input = xx(input)
        return input

    def build_query(
        self,
        folder=None,
        max_epochs=None,
        seeds=None, 
        devices=None,
        include_static_info=True,
        validate_data=True,
        others:dict =None
    ):
        folder = others.get('folder', self.NASBenchASR_DATAPATH)
        max_epochs = others.get('max_epochs', None)
        seeds = others.get('seeds', None)
        devices = others.get('devices', None)
        include_static_info = others.get('include_static_info', True)
        validate_data = others.get('validate_data', True)
        self._query = from_folder(folder, max_epochs, seeds, devices, include_static_info, validate_data)
        return self._query

    def is_dataset_ready(self):
        flag1 = os.path.exists(self.NASBenchASR_DATAPATH)
        if flag1:
            flag2 = len(os.listdir(self.NASBenchASR_DATAPATH)) > 0
            return flag1 and flag2
        return flag1

    def prepare_dataset(self):
        if not os.path.exists(self.NASBenchASR_DATAPATH):
            os.makedirs(self.NASBenchASR_DATAPATH)
        if len(os.listdir(self.NASBenchASR_DATAPATH)) <= 0:
            crt_file_path = os.path.abspath(__file__)
            crt_folder_path = os.path.dirname(crt_file_path)
            download_shell = os.path.join(crt_folder_path, 'download_nasbenchasr.sh')
            print('Downloading NASBenchASR dataset...')
            os.system(f"bash {download_shell}")
            print('Done')

    def query_by_key(self, key: str, **kwargs):
        if not self.is_dataset_ready():
            self.prepare_dataset()
        if key == 'full':
            return self.query_full_info(**kwargs)
        elif key == 'test_acc':
            return self.query_test_acc(**kwargs)
        elif key == 'val_acc':
            return self.query_val_acc(**kwargs)
        elif key == 'latency':
            return self.query_latency(**kwargs)
        elif key == 'params':
            return self.query_params(**kwargs)
        elif key == 'flops':
            return self.query_flops(**kwargs)
        else:
            raise NotImplementedError(f'{key} not supported.')

    def query_full_info(self, **kwargs):
        default_kwargs = {
            "arch": self.arch,
            "include_static_info": True,
            "seed":None,
            "devices":None,
            "include_static_info":None,
            "return_dict":True,
        }
        default_kwargs.update({key:value for key, value in kwargs.items() if key in default_kwargs})
        
        query = self.build_query(others=kwargs)
        return query.full_info(**default_kwargs)

    def query_test_acc(self, **kwargs):
        default_kwargs = {
            "arch": self.arch,
            "seed":None,
        }
        default_kwargs.update({key:value for key, value in kwargs.items() if key in default_kwargs})
        query = self.build_query(others=kwargs)
        return query.test_acc(**default_kwargs)

    def query_val_acc(self, **kwargs):
        default_kwargs = {
            "arch": self.arch,
            "epoch": None,
            "best": True,
            "seed":None,
        }
        default_kwargs.update({key:value for key, value in kwargs.items() if key in default_kwargs})
        query = self.build_query(others=kwargs)
        return query.val_acc(**default_kwargs)

    def query_latency(self, **kwargs):
        default_kwargs = {
            "arch": self.arch,
            "devices": None,
            "return_dict": True
        }
        default_kwargs.update({key:value for key, value in kwargs.items() if key in default_kwargs})
        query = self.build_query(others=kwargs)
        return query.latency(**default_kwargs)

    def query_params(self, **kwargs):
        default_kwargs = {
            "arch": self.arch
        }
        default_kwargs.update({key:value for key, value in kwargs.items() if key in default_kwargs})
        query = self.build_query(others=kwargs)
        return query.params(**default_kwargs)

    def query_flops(self, **kwargs):
        default_kwargs = {
            "arch": self.arch
        }
        default_kwargs.update({key:value for key, value in kwargs.items() if key in default_kwargs})
        query = self.build_query(others=kwargs)
        return query.flops(**default_kwargs)

    @property
    def arch(self):
        list_desc = self.dict_mask_to_list_desc(self.mask)
        for node_idx in range(len(list_desc)):
            name = list_desc[node_idx][0]
            list_desc[node_idx][0]= self.main_op_name2idx[name]
        return list_desc

    @classmethod
    def list_desc_to_dict_mask(self, list_desc):
        '''
        list_desc = [
            ['linear', 1],
            ['conv5', 1, 1],
            ['conv7d2', 1, 0, 1],
        ]
        '''
        mask = {}
        num_blocks = 4
        scells_per_block = [3, 4, 5, 6]
        num_nodes=3
        assert len(list_desc) == num_nodes, f"NASBenchASR fixes #node=3"

        for block_idx in range(num_blocks):
            for cell_idx in range(scells_per_block[block_idx]):
                for node_idx in range(len(list_desc)):
                    prefix = f"block{block_idx}_cell{cell_idx}_node{node_idx}"
                    for edge_idx in range(len(list_desc[node_idx])):
                        if edge_idx == 0:
                            main_key = f"{prefix}_main"
                            main_op_idx = self.main_op_name2idx[list_desc[node_idx][0]]
                            mask[main_key] = torch.eye(len(self.main_op_name2idx))[main_op_idx].bool()
                        else:
                            skip_key = f"{prefix}_skip{edge_idx-1}"
                            skip_op_idx = list_desc[node_idx][edge_idx]
                            mask[skip_key] = torch.eye(2)[skip_op_idx].bool()
        return mask

    @classmethod
    def dict_mask_to_list_desc(self, mask: dict):  
        list_desc = []  
    
        num_blocks = 4  
        scells_per_block = [3, 4, 5, 6]  
        num_nodes = 3  
    
        for block_idx in range(num_blocks):  
            for cell_idx in range(scells_per_block[block_idx]):
                for node_idx in range(num_nodes):
                    node_desc = []
                    prefix = f"block{block_idx}_cell{cell_idx}_node{node_idx}"
    
                    main_key = f"{prefix}_main"  
                    main_op_idx = torch.argmax(mask[main_key].float()).item()  
                    main_op_name = self.main_op_idx2name[main_op_idx]  
                    node_desc.append(main_op_name)
                    
                    for edge_idx in range(node_idx+1):  
                        skip_key = f"{prefix}_skip{edge_idx}"
                        skip_op_idx = torch.argmax(mask[skip_key].float()).item()  
                        node_desc.append(skip_op_idx)
                    list_desc.append(node_desc)
        return list_desc[:num_nodes] 


if __name__ == '__main__':
    from hyperbox.mutator import RandomMutator
    model = NASBenchASR()
    print(sum([p.numel() for p in model.parameters()]))
    rm = RandomMutator(model)
    rm.reset()
    B, F, T = 2, 80, 30
    for T in [16]:
        x = torch.rand(B, F, T)
        rm.reset()
        y = model(x)
        print(y.shape)
        # print(rm._cache, len(rm._cache))


    list_desc = [
        ['linear', 1],
        ['conv5', 1, 0],
        ['conv7d2', 1, 0, 1],
    ]
    mask = NASBenchASR.list_desc_to_dict_mask(list_desc)
    # print(mask)
    model2 = NASBenchASR(mask=mask)
    print(sum([p.numel() for p in model2.parameters()]))
    print(model2.arch_size((B, F, T)))
    y = model(x)
    print(NASBenchASR.dict_mask_to_list_desc(mask))

    # print(model2.query_full_info(max_epochs=5))
    # print(model2.query_flops())
    # print(model2.query_latency())
    # print(model2.query_params())
    # print(model2.query_test_acc())
    # print(model2.query_val_acc())

    for key in ['full', 'flops', 'test_acc', 'params', 'val_acc', 'latency']:
        print(key, model2.query_by_key(key))
