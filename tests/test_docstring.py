import doctest
from deepda import kalman_filter


def test_kalman_filter_docstrings():
    assert (
        doctest.testmod(kalman_filter).failed == 0
    ), "Failed docstring tests in deepda.kalman_filter :("
