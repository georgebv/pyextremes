import numpy as np
import pytest

from pyextremes.extremes import get_extremes
from pyextremes.extremes.block_maxima import NoDataBlockWarning


def test_invalid_arguments(battery_wl):
    # Test wrong extremes_type value
    with pytest.raises(ValueError, match=r"invalid value.*extremes_type"):
        get_extremes(
            ts=battery_wl,
            method="BM",
            extremes_type="BAD EXTREMES_TYPE VALUE",
            block_size="1Y",
            errors="coerce",
        )

    # Test wrong block_size type
    with pytest.raises(TypeError, match=r"invalid type.*block_size"):
        get_extremes(
            ts=battery_wl,
            method="BM",
            extremes_type="high",
            block_size=1,
            errors="coerce",
        )

    # Test wrong errors value
    with pytest.raises(ValueError, match=r"invalid value.*errors.*argument"):
        get_extremes(
            ts=battery_wl,
            method="BM",
            extremes_type="high",
            block_size="1Y",
            errors="BAD ERRORS VALUE",
        )


@pytest.mark.parametrize("extremes_type", ["high", "low"])
def test_extreme_value_extraction(battery_wl, extremes_type):
    # Test errors=raise
    with pytest.raises(ValueError):
        get_extremes(
            ts=battery_wl,
            method="BM",
            extremes_type=extremes_type,
            block_size="1Y",
            errors="raise",
        )

    # Test errors=ignore
    with pytest.warns(NoDataBlockWarning, match=r"blocks contained no data"):
        extremes_ignored = get_extremes(
            ts=battery_wl,
            method="BM",
            extremes_type=extremes_type,
            block_size="1Y",
            errors="ignore",
        )
        assert len(extremes_ignored) == 96

    # Test errors=coerce
    with pytest.warns(NoDataBlockWarning, match=r"blocks contained no data"):
        extremes_coerced = get_extremes(
            ts=battery_wl,
            method="BM",
            extremes_type=extremes_type,
            block_size="1Y",
            errors="coerce",
        )
        assert len(extremes_coerced) == 100

    if extremes_type == "high":
        assert np.isclose(extremes_ignored.max(), extremes_coerced.max())
    else:
        assert np.isclose(extremes_ignored.min(), extremes_coerced.min())
    assert np.isclose(extremes_ignored.mean(), extremes_coerced.mean())
