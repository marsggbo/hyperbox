
import torch
import torch.nn as nn

from pytorch_lightning.utilities.seed import seed_everything

from hyperbox.networks import OFAMobileNetV3, DartsNetwork, ENASMacroGeneralModel, ENASMicroNetwork, BaseNASNetwork
from hyperbox.mutator import RandomMutator
from hyperbox.utils.utils import load_json
from hyperbox.mutables.ops import Conv2d, BatchNorm2d, FinegrainedModule, Linear
from hyperbox.mutables.spaces import InputSpace, OperationSpace, ValueSpace
from hyperbox.utils.metrics import accuracy

def is_module_equal(m1, m2):
    count = 0
    for (name1, p1), (name2, p2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        loss = (p1-p2).abs().sum()
        if loss != 0:
            count += 1
            print(name1, name2)
    if count != 0:
        return False
    return True


class Net(BaseNASNetwork):
    def __init__(self, mask=None):
        from hyperbox.networks.darts import DartsCell
        super().__init__()
        ops1 = [
            nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(3,8,kernel_size=5,stride=1,padding=2),
        ]
        ops2 = [
            nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(3,8,kernel_size=5,stride=1,padding=2),
        ]
        self.candidate_op1 = OperationSpace(ops1, key='candidate1', mask=mask)
        self.candidate_op2 = OperationSpace(ops2, key='candidate2', mask=mask)
        num_nodes = 1
        out_channels = int(8 / num_nodes)
        self.cell = DartsCell(num_nodes, 8, 8, out_channels, False, False, mask=mask)
        self.input_op = InputSpace(n_candidates=3, n_chosen=1, key='input1', mask=mask)
        
        v1 = ValueSpace([1,2,4,8], key='v1', mask=mask)
        v2 = ValueSpace([1,2], key='v2', mask=mask)
        v3 = ValueSpace([1,3], key='v3', mask=mask)
        self.fop1 = Conv2d(8, v1, kernel_size=v3,stride=v2,padding=1,auto_padding=True)
        self.fop2 = Conv2d(v1,v1,3,1,1)
        self.fop3 = BatchNorm2d(v1)
        self.fc = Linear(v1, NUM_CLASSES, bias=False)

    def forward(self, x):
        bs = x.shape[0]
        out1 = self.candidate_op1(x)
        out2 = self.candidate_op2(x)
        out3 = self.cell(out1, out2)
        
        out = self.input_op([out1, out2, out3])
        out = self.fop1(out)
        out = self.fop2(out)
        out = self.fop3(out)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(bs,-1)
        out = self.fc(out)
        return out


NUM_CLASSES = 5
def test_case(net_cls, *args, **kwargs):
    # try:
    x = torch.rand(32,3,64,64)
    y = torch.randint(0, NUM_CLASSES, (32,))
    print("="*20)
    name = net_cls.__name__
    # print(f"{name} start")
    net = net_cls(*args, **kwargs)
    # print(f"{name} init pass")
    m = RandomMutator(net)
    m.reset()
    net.eval()
    y1 = net(x)
    acc1 = accuracy(y1,y)
    mask_file = f'{name}.json'
    m.save_arch(mask_file)
    # print(f"{name} save arch pass")
    mask = load_json(mask_file)
    subnet = net.build_subnet(mask)
    origin_subnet_state = subnet.state_dict()
    subnet.eval()
    # print(f"{name} build_subnet pass")

    # check whether load state dict successfully
    # way 1
    subnet.load_from_supernet(net.state_dict())
    # print(f"{name} load subnet from supernet pass")
    updated_subnet_state = subnet.state_dict()
    y2 = subnet(x)
    acc2 = accuracy(y2, y)
    subnet.load_state_dict(origin_subnet_state)
    y3 = subnet(x)
    acc3 = accuracy(y3, y)
    if acc1!=acc2 or acc1!=acc3 or acc2!=acc3:
        print(acc1, acc2, acc3)
        print(f"{name} wrong")
        for idx, op in enumerate([subnet.candidate_op1, subnet.candidate_op2]):
            idx += 1
            index = op.index
            print(f'c{idx}', is_module_equal(op, eval(f"net.candidate_op{idx}")[index]))
        for idx, node in enumerate(subnet.cell.mutable_ops):
            for jdx, op in enumerate(node.ops):
                index = op.index
                pop = net.cell.mutable_ops[idx].ops[jdx][index]
                print(f"node{idx}-op{jdx}", is_module_equal(op, pop))
        pass
    else:
        # print(f"{name} pass")
        pass


if __name__ == '__main__':
    for i in range(2):
        seed_everything(i+999)
        # test_case(Net)
        test_case(OFAMobileNetV3, num_classes=NUM_CLASSES)
        test_case(DartsNetwork, 3, 16, NUM_CLASSES, 1) # 'cells.0.mutable_ops.0.ops.0.choices.0.bn.running_mean'
        test_case(ENASMacroGeneralModel, num_classes=NUM_CLASSES) # layers.0.mutable.choices.0.conv.weight
        # test_case(ENASMicroNetwork, num_classes=NUM_CLASSES) # 'layers.0.nodes.0.cell_x.op_choice.choices.1.conv.depthwise.weight'