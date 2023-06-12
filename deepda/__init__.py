import contextlib
from pkg_resources import DistributionNotFound, get_distribution

from .forwardModel import *  # noqa
from .kalman_filter import *  # noqa
from .variational import *  # noqa

with contextlib.suppress(DistributionNotFound):
    __version__ = get_distribution(__name__).version
