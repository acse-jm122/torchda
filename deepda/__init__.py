"""
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serialization of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""
from contextlib import suppress
from enum import Enum, auto, unique
from importlib.metadata import PackageNotFoundError, version
from typing import TypeVar

from numpy import ndarray
from torch import Tensor

# Suppress PackageNotFoundError to handle missing package metadata
with suppress(PackageNotFoundError):
    __version__ = version(__name__)


@unique
class Algorithms(Enum):
    """Enumeration for various algorithms."""
    EnKF = auto()   # Ensemble Kalman Filter
    Var3D = auto()  # 3D Variational assimilation
    Var4D = auto()  # 4D Variational assimilation


@unique
class Device(Enum):
    """Enumeration for device types."""
    CPU = auto()
    GPU = auto()


# Define a type hint for generic tensors used in the package
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
