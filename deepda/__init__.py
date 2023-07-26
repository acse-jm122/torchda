from contextlib import suppress
from enum import Enum, auto, unique

from pkg_resources import DistributionNotFound, get_distribution


@unique
class Algorithms(Enum):
    EnKF = auto()
    Var3D = auto()
    Var4D = auto()


from .builder import *  # noqa
from .kalman_filter import *  # noqa
from .variational import *  # noqa

with suppress(DistributionNotFound):
    __version__ = get_distribution(__name__).version
