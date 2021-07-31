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

## 2021年6月19日22:06:06

- `Model`: print model info (size, flops, arch encoding) during the search stage
- `Mutator`: update both `self._choices` and `mutable.mask`
- update `arch` attribute of `DartsNetwork`

## 2021年6月19日22:39:11

- add `arch` attribute to MobileNet
- add yaml files of `MobileNet` and `Mobile3DNet`

## 2021年6月20日14:15:15

- update `sample_search`  func to `DartsModel`

## 2021年6月20日14:16:51

- update `arch` of `MobileNet` and `Mobile3DNet`

## 2021年6月21日20:30:33

- add `mask.json` of DartsNetwork found in the original paper
- add `visualize_DARTS_cell.py` to visualize both `normal` and `reduce` darts cells

## 2021年6月21日22:41:55

- fixbug: when `mask` is a str path of the mask file, failed to parse it to a `dict` object

## 2021年6月26日11:31:28

- update arch encoding of DartsNetwork
- refactor `RandomMutator`,`OnehotMutator`,`DartsMutator`

## 2021年6月26日11:32:29

- add `cutout` transform
- fix bug in `AlbumentationsTransforms`

## 2021年6月26日11:34:13

- reset default value in yaml files
    - disable `mask` path in `darts_network` yaml file
    - `patience=20` in `EarlyStopping` callback
- add `OnehotMutator` yaml file


## 2021年6月26日11:37:56

- add features to `BaseModel` (`models/base_model.py`)
    - `sample_search`: specify the way of sampling
    - `reset_seed`: 在分布式搜索时可以通过在每个进程调用该函数来达到**异步搜索**的效果，而且每个进程的搜索结果是不同的。
        反之，默认情况下所有进程采用相同的seed，所以采样的结果是一样的
- refactor `ClassifyModel`

## 2021年6月27日16:09:39

- 实现**同步**和**异步**的`sample_search`函数，并且支持搜索相同或者不同的网络结构
    - `is_sync`
        - 若为`True`就是同步，即只由rank 0进程进行采样，然后广播（broadcast）给其他进程
        - 若为`False`就是异步，即每个进程彼此之间独立采样
    - `is_net_parallel`
        - `True`: 每个进程的训练的结构**不一样**
        - `False`: 每个进程的训练的结构**一样**

- 四种情况

| `is_sync` | `is_net_parallel` | 效果 |
| --- | --- | ---|
| True | True | 只由rank0采样，并且采样N次，使得N个进程的模型各不相同 |
| True | False | 只由rank0采样，并且采样1次，使得N个进程的模型相同 |
| False | True | 进程之间彼此独立采样，且模型不同 （每个进程需要重新设置种子数） |
| False | False | 进程之间彼此独立采样，且模型相同 （种子数都一样） |

- `RandomMutator`在`OFAModel`里调试通过
- `DartsMutator`无论是同步还是异步都只支持`is_net_parallel=False`，所以尽量使用异步，这样可以减少同步操作带来的broadcast消耗
- `OnehotMutator`
    - 同样也建议使用**异步搜索**。异步的情况下支持搜索相同或者不同的模型结构。
    - **如果是同步操作，只支持搜索相同的模型结构**，因为rank0的计算图无法同步到其他进程

## 2021年6月27日16:11:38

- 将`gradient_clip_val`的默认值设置为0，因为当该值大于0时，无法调用`LightningModule.manual_backward`方法

## 2021年6月27日16:40:17

- 统一Model类下的`sample_search`接口
- rename `random_nas_model.py` to `random_model.py`

## 2021年7月16日16:30:19

- add searchable layers2D
- 将torchmetrics计算Accuracy替换成自定义的模块，原来的比较耗时
- add `OFAMobileNetV3`
- update `ops.Conv`算子

## 2021年7月17日14:34:34

- benchmark OFA

# 2021年7月17日23:02:29

- add `scheduler_cfg` to `base_model`
- add `Lamb` optimizer and its yaml config

# 2021年7月20日14:19:54

- add `ImageNetDatamodule` and its yaml config
- fixbugs in `layers2d.py` and `conv.py`
- add `GradualWarmupScheduler` and its yaml config
- add yaml config of `Lamb` optimizer
- add `_module_available`
- add `scheduler_cfg` for `BaseModel` class
- update `GradualWarmupScheduler`

# 2021年7月20日17:17:07

- fix kernel size for `ofa_mbv3.py`
- `OFAModel`支持DALI数据集
```python
...
    def training_step(self, batch: Any, batch_idx: int):
        image, label = batch[0]['data'], batch[0]['label'].long().view(-1)
```

# 2021年7月21日16:54:07

- fixbug in `GradualWarmupScheduler`: epoch=0时， lr为0
- fixbug in `CrossEntropyLabelSmooth` & add yaml configure

# 2021年7月25日14:04:46

- update `DataModule` and their yaml configs

# 2021年7月27日00:12:33

- fix bug in `fixed_mutator`，适配 `ValueSpace`
- `load_json` 函数转化成 `torch.tensor`格式
- import predefined networks in `__init__`

# 2021年7月27日23:57:00

- 给所有模型增加`mask`参数，方便得到指定的子模型
- 更新`OperationSpace`的内置函数，以`choices`为准
- 修复`InputSpace`在`is_search=False`情况下的bug
- 更新基类`BaseNASNetwork`的`build_subnet`和`load_subnet_state_dict`函数
    TODO: 
    - [x] 有的子模型虽然能成功load父模型的权重，但是最后的结果会有偏差，但是还未找到原因

# 2021年7月28日19:23:08

- 修复子模型load父模型权重后结果会出现偏差的问题，原因是`InputSpace`在`is_search=False`时的`forward`函数存在逻辑欠缺，之前没有考虑了输出多个值的情况，导致有的结果就漏掉了。解决办法是参照`DefaultMutator`中的方法改进了

原来的代码
```python
class InputSpace(...):
    ...
    def forward(self, ...)
        if is_search:
            ...
        else:
            if isinstance(optional_inputs, list):
                index = self.index
            elif isinstance(optional_inputs, dict):
                index = self.choose_from[self.index]
            out = optional_inputs[index] # 这里只输出某一个值，但是像DARTS和ENAS可能会输出多个节点的值
```
- rename `load_subnet_state_dict` to `load_from_supernet`

# 2021年7月28日23:14:32

- fixbugs in `hyperbox.mutables.layers.layers2d`. 主要是通道数转化问题
- fixbugs in `calc_model_size.py`. 当`groups`是`ValueSpace`时，未正常取值
- remove `export_model_from_mask` in `BaseNASNetwork`，这个和`build_subnet`功能重叠

# 2021年7月31日19:06:48

- fixbugs in `OFAMobileNetV3`:
    - `kernel_size_list=[3,5,7]` by defeault
    - `self.runtime_depth`

# TODO

- [ ] 可视化模型结构
- [ ] 尝试，是的，尝试复现Once-for-all
- [ ] 实现`set_running_statistics`
- [ ] 目前的`spaces`编号方式是`global_counting`，这样不方便直接根据传入的`mask`得到指定模型结构，需要改进这种编码方式
- [x] NAS model能够自动导出其当前
  - [x] 结构编码信息: `self.arch`
  - [x] 当前模型大小: `self.model_size`
- [ ] 实现新的Callbacks
    - [ ] `ArchitectureCheckpoint`: 以`json`文件的格式导出搜索到的模型结构，
    - [ ] `ArchitectureHistory`： 记录搜索过程中模型结构以及对应的性能
- [x] 改进`sample_search`接口，支持以下功能
  - [x] **异步/同步**搜索
  - [x] 多个进程同时搜索 **相同/不同** 模型结构
  - [ ] 可扩展性强，能够轻松实现不同搜索