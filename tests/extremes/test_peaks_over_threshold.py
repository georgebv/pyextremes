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
            r="24H",
        )

    # Test wrong r type
    with pytest.raises(TypeError, match=r"invalid type.*'r' argument"):
        get_extremes(
            ts=battery_wl, method="POT", extremes_type="high", threshold=2, r=1
        )


@pytest.mark.parametrize("extremes_type, threshold", [("high", 1.35), ("low", -1.65)])
def test_extreme_value_extraction(battery_wl, extremes_type, threshold):
    extremes = get_extremes(
        ts=battery_wl,
        method="POT",
        extremes_type=extremes_type,
        threshold=threshold,
        r="24H",
    )
    if extremes_type == "high":
        assert np.isclose(extremes.max(), battery_wl.max())
        assert len(extremes) == 127
    elif extremes_type == "low":
        assert np.isclose(extremes.min(), battery_wl.min())
        assert len(extremes) == 109
    assert np.all(np.diff(extremes.index) > pd.to_timedelta("24H").to_numpy())
