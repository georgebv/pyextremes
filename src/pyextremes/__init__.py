__version__ = "2.4.0"
__all__ = [
    "EVA",
    "get_extremes",
    "get_model",
    "get_return_periods",
    "plot_mean_residual_life",
    "plot_parameter_stability",
    "plot_return_value_stability",
    "plot_threshold_stability",
]

from pyextremes.eva import EVA
from pyextremes.extremes import get_extremes, get_return_periods
from pyextremes.models import get_model
from pyextremes.tuning import (
    plot_mean_residual_life,
    plot_parameter_stability,
    plot_return_value_stability,
    plot_threshold_stability,
)
