from pyextremes.eva import EVA
from pyextremes.extremes import get_extremes, get_return_periods
from pyextremes.tuning import (
    plot_mean_residual_life,
    plot_parameter_stability,
    plot_return_value_stability,
)

__version__ = "1.0.2"
__all__ = [
    "EVA",
    "get_extremes",
    "get_return_periods",
    "plot_mean_residual_life",
    "plot_parameter_stability",
    "plot_return_value_stability",
]
