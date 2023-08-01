from contextlib import suppress
from enum import Enum, auto, unique
from importlib.metadata import PackageNotFoundError, version
from typing import TypeVar

from numpy import ndarray
from torch import Tensor


@unique
class Algorithms(Enum):
    EnKF = auto()
    Var3D = auto()
    Var4D = auto()


@unique
class Device(Enum):
    CPU = auto()
    GPU = auto()


_GenericTensor = TypeVar(
    "_GenericTensor",
    bound=list | tuple | ndarray | Tensor,
)  # noqa


from .builder import *  # noqa
from .kalman_filter import *  # noqa
from .variational import *  # noqa

with suppress(PackageNotFoundError):
    __version__ = version(__name__)
