import doctest
from deepda import builder, kalman_filter, variational


def test_builder_docstrings():
    assert (
        doctest.testmod(builder).failed == 0
    ), "Failed docstring tests in deepda.builder :("


def test_kalman_filter_docstrings():
    assert (
        doctest.testmod(kalman_filter).failed == 0
    ), "Failed docstring tests in deepda.kalman_filter :("


def test_variational_docstrings():
    assert (
        doctest.testmod(variational).failed == 0
    ), "Failed docstring tests in deepda.variational :("
