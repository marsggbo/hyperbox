<div align="center">

# Hyperbox


<a href="https://github.com/marsggbo/hyperbox"><img alt="Hyperbox" src="https://img.shields.io/badge/-Hyperbox-organe?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
<!-- <a href="https://hub.docker.com/r/ashlev/lightning-hydra"><img alt="Docker" src="https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white"></a> -->

A clean and scalable template to kickstart your AutoML project 🚀⚡🔥<br>
<!-- Click on [<kbd>Use this template</kbd>](https://github.com/ashleve/lightning-hydra-template/generate) to initialize new repository. -->

*Currently uses dev version of Hydra.<br>Suggestions are always welcome!*

</div>

You can refer to [Wiki](https://github.com/marsggbo/hyperbox/wiki) for more details.


<details>
<summary><b> Install </b></summary>

- install via `pip`
```
pip install hyperbox
```
- install via `github`

```
git clone https://github.com/marsggbo/hyperbox
cd hyperbox
python setup.py develop
python install -r requirements.txt
```

</details>

<details>
<summary><b> Quick Start </b></summary>

```
python -m hyperbox.run experiment=example_random_nas +trainer.fast_dev_run=True
```

</details>


<details>
<summary><b> Hyperbox Mutables (Searchable `nn.Module`) </b></summary>

<img width="806" alt="image" src="https://user-images.githubusercontent.com/13477956/203073104-9e2bdf61-e7e2-498b-9efb-d4fb0c096b95.png">

- Code implementation for Figure (left)

```python
import torch.nn as nn
from hyperbox . mutables . spaces import OperationSpace
op1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride =1, padding=1)
op2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride =1, padding=2)
op3 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride =1, padding=3)
ops = OperationSpace(candidates=[op1, op2, op3], key=’conv_op’, mask=[1, 0, 0])
```

Code implementation for Figure (middle)
```python
import torch
from hyperbox.mutables.spaces import InputSpace
in1 = torch.rand(2, 64)
in2 = torch.rand(2, 32)
in3 = torch.rand(2, 16)
inputs = [in1, in2, in3]
skipconnect = InputSpace(n_candidates=3, n_chosen=1, key=’sc’, mask=[0, 1, 0])
out = skipconnect([ in1 , in2 , in3 ]) 
assert out is in2
>>> True
```

- Code implementation for Figure (right)

```python
from hyperbox.mutables.ops import Conv2d, Linear 
from hyperbox.mutables.spaces import ValueSpace

# convolution
ks = ValueSpace([3, 5, 7], key=’kernelSize’, mask=[0, 1, 0])
cout = ValueSpace([16, 32, 64], key=’channelOut’, mask=[0, 1, 0])
conv = Conv2d(3 , cout , ks , stride =1, padding=2, bias=False )
print([x.shape for x in conv.parameters()])
>>> [torch.Size([32, 3, 5, 5])]

# linear
cout = ValueSpace([10, 100], key=’channelOut1’)
isBias = ValueSpace([0, 1], key=’bias’) # 0: False, 1: True 
linear = Linear(10, cout , bias=isBias)
print([x.shape for x in linear.parameters()])
>>> [ torch . Size ([100 , 10]) , torch . Size ([100]) ]
```

</details>

<details>
<summary><b> Hyperbox Mutator (Search Algorithms) </b></summary>

- Random Search Algorithm `RandomMutator`

```python
from hyperbox.mutator import RandomMutator
from hyperbox.networks.ofa import OFAMobileNetV3

net = OFAMobileNetV3()
rm = RandomMutator(net)
rm.reset() # search a subnet
arch: dict = rm._cache # arch of a subnet
print(arch)

subnet = OFAMobileNetV3(mask=arch) # initialize a subnet, which has smaller parameters than `net`
```

the model arch is saved in `rm._cache`

</details>


<details>
<summary><b> Hyperbox Wikis </b></summary>

- [Wiki for hyperbox.config](https://github.com/marsggbo/hyperbox/wiki/Customize-Config)
- [Wiki for hyperbox.mutables](https://github.com/marsggbo/hyperbox/wiki/Customize-Mutable)
- [Wiki for hyperbox.engine](https://github.com/marsggbo/hyperbox/wiki/Customize-Engine)
- [Wiki for hyperbox.mutator](https://github.com/marsggbo/hyperbox/wiki/Customize-Mutator)
- [Wiki for hyperbox.models](https://github.com/marsggbo/hyperbox/wiki/Customize-Models)
- [Wiki for hyperbox.networks](https://github.com/marsggbo/hyperbox/wiki/Customize-NAS-Network)
- [Wiki for Hydra](https://github.com/marsggbo/hyperbox/wiki/Hydra-Q&A)
- [Wiki for Hyperbox App](https://github.com/marsggbo/hyperbox/wiki/Hyperbox-App:-Start-a-new-project)
- [Miscellaneous](https://github.com/marsggbo/hyperbox/wiki/Miscellaneous-(tricks))
- [Q&A](https://github.com/marsggbo/hyperbox/wiki/Q&A)
- [Usage](https://github.com/marsggbo/hyperbox/wiki/Usages)

</details>


## Thanks

[![](https://shields.io/badge/-NNI-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/microsoft/nni/tree/v1.7)


[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)   
