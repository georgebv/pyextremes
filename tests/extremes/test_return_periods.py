import numpy as np
import pytest

from pyextremes.extremes import get_return_periods


def test_get_return_periods_errors(battery_wl_preprocessed, extremes_bm_high):
    # Test bad block_size type
    with pytest.raises(TypeError, match=r"invalid type.*block_size.*argument"):
        get_return_periods(
            ts=battery_wl_preprocessed,
            extremes=extremes_bm_high,
            extremes_method="BM",
            extremes_type="high",
            block_size=1,
            return_period_size="1Y",
            plotting_position="weibull",
        )

    # Test block_size for POT
    with pytest.raises(ValueError, match=r"block_size.*only if 'extremes_method'"):
        get_return_periods(
            ts=battery_wl_preprocessed,
            extremes=extremes_bm_high,
            extremes_method="POT",
            extremes_type="high",
            block_size="1Y",
            return_period_size="1Y",
            plotting_position="weibull",
        )

    # Test bad return_period_size type
    with pytest.raises(TypeError, match=r"invalid type.*return_period_size"):
        get_return_periods(
            ts=battery_wl_preprocessed,
            extremes=extremes_bm_high,
            extremes_method="BM",
            extremes_type="high",
            block_size="1Y",
            return_period_size=1,
            plotting_position="weibull",
        )

    # Test bad extremes_method
    with pytest.raises(ValueError, match=r"invalid value.*extremes_method"):
        get_return_periods(
            ts=battery_wl_preprocessed,
            extremes=extremes_bm_high,
            extremes_method="BAD EXTREMES_METHOD VALUE",
            extremes_type="high",
            block_size=None,
            return_period_size="1Y",
            plotting_position="weibull",
        )

    # Test bad extremes_type
    with pytest.raises(ValueError, match=r"invalid value.*extremes_type"):
        get_return_periods(
            ts=battery_wl_preprocessed,
            extremes=extremes_bm_high,
            extremes_method="BM",
            extremes_type="BAD EXTREMES_TYPE VALUE",
            block_size="1Y",
            return_period_size="1Y",
            plotting_position="weibull",
        )

    # Test bad plotting_position
    with pytest.raises(ValueError, match=r"invalid value.*plotting_position"):
        get_return_periods(
            ts=battery_wl_preprocessed,
            extremes=extremes_bm_high,
            extremes_method="BM",
            extremes_type="high",
            block_size="1Y",
            return_period_size="1Y",
            plotting_position="BAD PLOTTING_POSITION VALUE",
        )

    # Test automatic block_size type
    return_periods_automatic = get_return_periods(
        ts=battery_wl_preprocessed,
        extremes=extremes_bm_high,
        extremes_method="BM",
        extremes_type="high",
        block_size=None,
        return_period_size="1Y",
        plotting_position="weibull",
    )
    return_periods = get_return_periods(
        ts=battery_wl_preprocessed,
        extremes=extremes_bm_high,
        extremes_method="BM",
        extremes_type="high",
        block_size="1Y",
        return_period_size="1Y",
        plotting_position="weibull",
    )
    assert (
        np.abs(
            return_periods_automatic.loc[:, "return period"].values.max()
            - return_periods.loc[:, "return period"].values.max()
        )
        <= 1
    )


@pytest.mark.parametrize("extremes_type", ["high", "low"])
def test_extremes_method_bm(
    battery_wl_preprocessed, extremes_bm_high, extremes_bm_low, extremes_type
):
    if extremes_type == "high":
        extremes = extremes_bm_high
    else:
        extremes = extremes_bm_low

    for plotting_position in [
        "ecdf",
        "hazen",
        "weibull",
        "tukey",
        "blom",
        "median",
        "cunnane",
        "gringorten",
        "beard",
    ]:
        return_periods = get_return_periods(
            ts=battery_wl_preprocessed,
            extremes=extremes,
            extremes_method="BM",
            extremes_type=extremes_type,
            block_size="1Y",
            return_period_size="1Y",
            plotting_position=plotting_position,
        )
        if extremes_type == "high":
            assert np.argmax(return_periods.loc[:, extremes.name].values) == np.argmax(
                return_periods.loc[:, "return period"].values
            )
        else:
            assert np.argmin(return_periods.loc[:, extremes.name].values) == np.argmax(
                return_periods.loc[:, "return period"].values
            )


@pytest.mark.parametrize("extremes_type", ["high", "low"])
def test_extremes_method_pot(
    battery_wl_preprocessed, extremes_pot_high, extremes_pot_low, extremes_type
):
    if extremes_type == "high":
        extremes = extremes_pot_high
    else:
        extremes = extremes_pot_low

    for plotting_position in [
        "ecdf",
        "hazen",
        "weibull",
        "tukey",
        "blom",
        "median",
        "cunnane",
        "gringorten",
        "beard",
    ]:
        return_periods = get_return_periods(
            ts=battery_wl_preprocessed,
            extremes=extremes,
            extremes_method="POT",
            extremes_type=extremes_type,
            block_size=None,
            return_period_size="1Y",
            plotting_position=plotting_position,
        )
        if extremes_type == "high":
            assert np.argmax(return_periods.loc[:, extremes.name].values) == np.argmax(
                return_periods.loc[:, "return period"].values
            )
        else:
            assert np.argmin(return_periods.loc[:, extremes.name].values) == np.argmax(
                return_periods.loc[:, "return period"].values
            )
