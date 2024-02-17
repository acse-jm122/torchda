"""
TorchDA
=======

Use Deep Learning in Data Assimilation

TorchDA is a Python package that provides a flexible and user-friendly
framework for performing data assimilation with neural networks on
various algorithms, including Ensemble Kalman Filter (EnKF),
3D Variational (3D-Var) assimilation, and
4D Variational (4D-Var) assimilation.

This package is designed to simplify the process of configuring and
executing data assimilation cases, making it easier for researchers
and practitioners to apply data assimilation techniques with
neural networks to their scientific and engineering problems.

Modules
-------
- parameters:
    Contains the Parameters class for specifying data assimilation parameters.
- builder:
    Provides a CaseBuilder class for configuring and
    executing data assimilation cases.
- executor:
    Implements the _Executor class for executing data assimilation cases.
- kalman_filter:
    Implements EnKF and Kalman Filter algorithms for data assimilation.
- variational:
    Implements 3D-Var and 4D-Var algorithms for variational data assimilation.

For more information, please refer to the package documentation.
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

    EnKF = auto()  # Ensemble Kalman Filter
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


from .builder import CaseBuilder  # noqa
from .executor import _Executor  # noqa
from .kalman_filter import apply_EnKF, apply_KF  # noqa
from .parameters import Parameters  # noqa
from .variational import apply_3DVar, apply_4DVar  # noqa

__all__ = (
    "Parameters",
    "CaseBuilder",
    "apply_KF",
    "apply_EnKF",
    "apply_3DVar",
    "apply_4DVar",
    "_Executor",
)
