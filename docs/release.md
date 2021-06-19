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
    - 统一`mutator`和`networks`模块中对`mutables`调用的方法，即`from hyperbox.mutables.spaces import InputSpace, *`
    - 重构`mobilenet`各个模块

## 2021年6月13日21:41:52

- refactor测试模块： 将原来写在代码下的测试模块抽离到`tests`目录下

## 2021年6月13日21:56:30

- rename `mutables.mutables` to `mutables.spaces`

## 2021年6月13日22:29:30

- add `auto_padding` feature to `mutables.ops.Conv1/2/3d`

## 2021年6月15日12:00:44

- 给`Mutable`增加`is_search`属性，传入`mask`后可以不通过`Mutator`直接运行`forward`函数
- 更新`Mutables`的`__repr__`
- 增加`is_searchable`函数用于判断`Mutable`的属性`is_search`是否为`True`

## 2021年6月15日14:03:01

- 修复计算模型大小和FLOPs时的bug

## 2021年6月15日14:41:18

- 支持传统的图像分类模型`ClassifyModel`

## 2021年6月16日21:59:25

- fix `flops_size_counter` to support calcutate `FinegrainedModule` flops and size
- add `arch_size` function for `BaseModel`
- refactor `ops.Conv / BatchNorm / Linear`
- add test modules for `ops.Conv / BatchNorm / Linear`

## 2021年6月17日11:31:20

- add `arch_size` function for `BaseNASNetwork`

## 2021年6月19日12:21:58

- remove deprecated `cfg` attribute in `Mutator`s

## 2021年6月19日14:10:07

- 重构 `model` config文件配置,示例如下。`network_cfg`、`mutator_cfg`等都设置单独的yaml文件，提高复用性

```yaml
_target_: hyperbox.models.ofa_model.OFAModel

defaults:
    - network_cfg: finegrained_resnet
    - mutator_cfg: random_mutator
    - optimizer_cfg: adam
    - metric_cfg: accuracy
    - loss_cfg: cross_entropy
```

## 2021年6月19日17:27:12

- add `transforms` to `DataModule`

# TODO

- [ ] 可视化模型结构
- [ ] 尝试，是的，尝试复现Once-for-all
- [ ] 实现`set_running_statistics`
- [ ] 目前的`spaces`编号方式是`global_counting`，这样不方便直接根据传入的`mask`得到指定模型结构，需要改进这种编码方式
- [ ] NAS model能够自动导出其当前
  - [ ] 结构编码信息: `self.arch`
  - [x] 当前模型大小: `self.model_size`
- [ ] 实现新的Callbacks
    - [ ] `ArchitectureCheckpoint`: 以`json`文件的格式导出搜索到的模型结构，
    - [ ] `ArchitectureHistory`： 记录搜索过程中模型结构以及对应的性能