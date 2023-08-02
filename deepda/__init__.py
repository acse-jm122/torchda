from contextlib import suppress
from enum import Enum, auto, unique
from importlib.metadata import PackageNotFoundError, version
from typing import TypeVar

from numpy import ndarray
from torch import Tensor

with suppress(PackageNotFoundError):
    __version__ = version(__name__)


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


from .builder import CaseBuilder, Parameters  # noqa
from .kalman_filter import apply_EnKF, apply_KF  # noqa
from .variational import apply_3DVar, apply_4DVar  # noqa

__all__ = (
    "Parameters",
    "CaseBuilder",
    "apply_KF",
    "apply_EnKF",
    "apply_3DVar",
    "apply_4DVar",
)
