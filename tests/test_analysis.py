import pytest
import pandas as pd

from ra_task.analysis import annual_summary, quadratic_weighted_kappa, wilson_interval


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


def test_annual_summary_uses_each_years_denominator_and_three_thresholds() -> None:
    ads = pd.DataFrame({"canonical_id": ["1", "2", "3", "4"], "year": [2024, 2024, 2025, 2025]})
    labels = pd.DataFrame({"canonical_id": ["1", "2", "3", "4"], "score": [0, 2, 1, 3]})
    result = annual_summary(ads, labels).set_index("year")
    assert result.loc[2024, "n_ads"] == 2
    assert result.loc[2024, "share_score_ge_2"] == pytest.approx(0.5)
    assert result.loc[2025, "share_score_ge_1"] == pytest.approx(1.0)
    assert result.loc[2025, "share_score_eq_3"] == pytest.approx(0.5)
