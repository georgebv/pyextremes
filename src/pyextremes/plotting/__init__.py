__all__ = [
    "plot_corner",
    "plot_extremes",
    "plot_probability",
    "plot_return_values",
    "plot_trace",
    "pyextremes_rc",
]

from pyextremes.plotting.extremes import plot_extremes
from pyextremes.plotting.mcmc import plot_corner, plot_trace
from pyextremes.plotting.probability_plots import plot_probability
from pyextremes.plotting.return_values import plot_return_values
from pyextremes.plotting.style import pyextremes_rc
