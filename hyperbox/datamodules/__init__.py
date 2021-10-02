from .cifar_datamodule import *
from .fakedata_datamodule import *
from .mnist_datamodule import *

from hyperbox.utils.utils import _module_available
if _module_available("nvidia.dali"):
    from .imagenet_datamodule import *
