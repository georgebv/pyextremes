import pytest

from pyextremes.plotting import plot_extremes


def test_plot_extremes_errors(battery_wl):
    # Test bad extremes_type for POT
    with pytest.raises(ValueError):
        plot_extremes(
            ts=battery_wl,
            extremes=battery_wl,
            extremes_method="POT",
            extremes_type="BAD TYPE",
            block_size="1Y",
            figsize=(8, 5),
        )

    # Test bad block_size for BM
    with pytest.raises(TypeError):
        plot_extremes(
            ts=battery_wl,
            extremes=battery_wl,
            extremes_method="BM",
            extremes_type="high",
            block_size=1,
            figsize=(8, 5),
        )

    # Test bad extremes_method
    with pytest.raises(ValueError):
        plot_extremes(
            ts=battery_wl,
            extremes=battery_wl,
            extremes_method="BAD METHOD",
            extremes_type="high",
            block_size="1Y",
            figsize=(8, 5),
        )
