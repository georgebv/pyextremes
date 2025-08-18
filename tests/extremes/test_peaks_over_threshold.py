import numpy as np
import pandas as pd
import pytest

from pyextremes.extremes import get_extremes


def test_invalid_arguments(battery_wl):
    # Test wrong extremes_type value
    with pytest.raises(ValueError, match=r"invalid value.*extremes_type"):
        get_extremes(
            ts=battery_wl,
            method="POT",
            extremes_type="BAD EXTREMES_TYPE VALUE",
            threshold=2,
            r="24h",
        )

    # Test wrong r type
    with pytest.raises(ValueError, match=r"invalid value.*'r' argument"):
        get_extremes(
            ts=battery_wl, method="POT", extremes_type="high", threshold=2, r="abc"
        )


@pytest.mark.parametrize("extremes_type, threshold", [("high", 1.35), ("low", -1.65)])
def test_extreme_value_extraction(battery_wl, extremes_type, threshold):
    extremes = get_extremes(
        ts=battery_wl,
        method="POT",
        extremes_type=extremes_type,
        threshold=threshold,
        r="24h",
    )
    if extremes_type == "high":
        assert np.isclose(extremes.max(), battery_wl.max())
        assert len(extremes) == 117
    elif extremes_type == "low":
        assert np.isclose(extremes.min(), battery_wl.min())
        assert len(extremes) == 104
    assert np.all(np.diff(extremes.index) > pd.to_timedelta("24h").to_numpy())


def test_single_cluster():
    index = pd.date_range(start="2021/01/01", end="2021/01/02", freq="1h")
    data = pd.Series(data=np.random.random(size=len(index)), index=index)

    # Tesh high extremes
    extremes = get_extremes(
        ts=data,
        method="POT",
        extremes_type="high",
        threshold=data.min() - 1,
        r="24h",
    )
    assert len(extremes) == 1
    assert np.isclose(extremes.values[0], data.max())

    # Tesh low extremes
    extremes = get_extremes(
        ts=data,
        method="POT",
        extremes_type="low",
        threshold=data.max() + 1,
        r="24h",
    )
    assert len(extremes) == 1
    assert np.isclose(extremes.values[0], data.min())


@pytest.mark.parametrize("extremes_type", ["high", "low"])
def test_threshold_producing_empty_series(battery_wl: pd.Series, extremes_type: str):
    """
    Regression test for scenario when threshold value is higher than the maximum value
    for extremes_type="high" and lower than the minimum value for extremes_type="low"
    in the time series. Original implementation would throw a ValueError when trying
    to get the index of the maximum/minium value in the resulting series because it
    would be empty.

    """
    with pytest.warns(UserWarning, match=r"^Threshold value.+zero extreme values$"):
        extremes = get_extremes(
            ts=battery_wl,
            method="POT",
            extremes_type=extremes_type,
            threshold=battery_wl.max() + 1
            if extremes_type == "high"
            else battery_wl.min() - 1,
            r="24h",
        )
    assert isinstance(extremes, pd.Series)
    assert len(extremes) == 0
    assert extremes.dtype == np.float64
    assert extremes.index.name == battery_wl.index.name


def test_generate_clusters_single_exceedance():
    index = pd.to_datetime(["2021-01-01 00:00:00"])
    data = pd.Series([42.0], index=index)
    extremes = get_extremes(
        ts=data,
        method="POT",
        extremes_type="high",
        threshold=41.0,
        r="24h",
    )
    assert len(extremes) == 1
    assert extremes.iloc[0] == 42.0
    assert extremes.index[0] == index[0]
