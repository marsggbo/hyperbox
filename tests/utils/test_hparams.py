import torch

from hyperbox.utils.utils import hparams_wrapper

@hparams_wrapper
class MyClass1(torch.nn.Module):
    def __init__(self, arg1):
        super(MyClass1, self).__init__()
        self.arg1 = arg1
        self.linear = torch.nn.Linear(3, arg1)

class MyClass2(MyClass1):
    def __init__(self, arg1, arg2, arg3=False):
        super(MyClass2, self).__init__(arg1)
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.conv = torch.nn.Conv2d(3, arg1, arg2, bias=arg3)
    
    def __setattr__(self, name, value):
        if name != 'key':
            super(MyClass2, self).__setattr__(name, value)


if __name__ == '__main__':
    print(MyClass1.__mro__) # (<class '__main__.MyClass1'>, <class 'torch.nn.modules.module.Module'>, <class 'object'>)
    print(MyClass2.__mro__) # (<class '__main__.MyClass2'>, <class '__main__.MyClass1'>, <class 'torch.nn.modules.module.Module'>, <class 'object'>)
    # Test 1
    obj1 = MyClass1(1)
    print(obj1.hparams)  # Output: {'arg1': 1}
    print(obj1)  # Output: 
    # MyClass1(
    #     (linear): Linear(in_features=3, out_features=1, bias=True)
    # )

    # Test 2
    obj2 = MyClass2(3,4)
    print(obj2.hparams)  # Output: {'arg3': False, 'arg1': 3, 'arg2': 4}
    print(obj2)  # Output: 
    # MyClass2(
    # (conv): Conv2d(3, 3, kernel_size=(4, 4), stride=(1, 1), bias=False)
    # )

    # Test 3
    from hyperbox.mutables.spaces import OperationSpace
    obj3 = OperationSpace([torch.nn.Linear(2,3),torch.nn.Linear(3,2)])
    print(obj3.hparams)  # Output: {'mask': None, 'index': None, 'reduction': 'sum', 'return_mask': False, 'key': None, 'candidates': [Linear(in_features=2, out_features=3, bias=True), Linear(in_features=3, out_features=2, bias=True)]}
    print(obj3)  # Output:
    # OperationSpace(
    # (candidates): ModuleList(
    #     (0): Linear(in_features=2, out_features=3, bias=True)
    #     (1): Linear(in_features=3, out_features=2, bias=True)
    # )
    # )
