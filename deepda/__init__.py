from contextlib import suppress
from enum import Enum, auto, unique
from importlib.metadata import PackageNotFoundError, version


@unique
class Algorithms(Enum):
    EnKF = auto()
    Var3D = auto()
    Var4D = auto()


@unique
class Device(Enum):
    CPU = auto()
    GPU = auto()


from .builder import *  # noqa
from .kalman_filter import *  # noqa
from .variational import *  # noqa

with suppress(PackageNotFoundError):
    __version__ = version(__name__)
