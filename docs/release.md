# Release Notes

## 2021年6月13日14:54:44

- rename `*Choice*` to `*Space*`
    - `LayerChoice` -> `OperationSpace`
    - `InputChoice` -> `InputSpace`
    - `ValueChoice` -> `ValueSpace`

## 2021年6月13日21:33:50

- refactor `mutables`
    - 将`finegrained_ops`重构成`ops`模块
    - 将原先的各种`Finegrained*`算子的名字与Pytorch保持一致，方便使用,例如
        - `FinegrainedConv2d` -> `Conv2d`
        - `FinegrainedBN2d` -> `BatchNorm2d`
    - 统一`mutator`和`networks`模块中对`mutables`调用的方法，即`from hyperbox.mutables.mutables import InputSpace, *`
    - 重构`mobilenet`各个模块

## 2021年6月13日21:41:52

- refactor测试模块： 将原来写在代码下的测试模块抽离到`tests`目录下