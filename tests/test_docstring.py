import doctest
from deepda import EnKF


def test_myEnKF_docstrings():
    assert (
        doctest.testmod(EnKF).failed == 0
    ), "Failed docstring tests in deepda.EnKF :("
