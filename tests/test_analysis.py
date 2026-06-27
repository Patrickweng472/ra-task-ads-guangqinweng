import pytest

from ra_task.analysis import quadratic_weighted_kappa, wilson_interval


def test_wilson_interval_boundaries() -> None:
    low, high = wilson_interval(0, 10)
    assert low == 0
    assert 0 < high < 0.5
    low, high = wilson_interval(10, 10)
    assert 0.5 < low < 1
    assert high == 1


def test_weighted_kappa() -> None:
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1)
    assert quadratic_weighted_kappa([0, 0, 3, 3], [3, 3, 0, 0]) < 0

