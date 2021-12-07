from hyperbox.utils.utils import _module_available
from .cifar_datamodule import *
from .fakedata_datamodule import *
from .mnist_datamodule import *

if _module_available("medmnist"):
    from .medmnist_datamodule import *
if _module_available("nvidia.dali"):
    from .imagenet_datamodule import *
