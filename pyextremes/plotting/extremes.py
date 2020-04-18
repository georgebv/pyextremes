import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import typing

from pyextremes.plotting.style import pyextremes_rc


def plot_extremes(
        extremes: pd.Series,
        ts: pd.Series = None,
        figsize: typing.Union[tuple, list] = (8, 5)
):
    """
    Plot time series of extreme events.

    Parameters
    ----------
    extremes : pandas.Series
        Time series of extreme events.
    ts : pandas.Series, optional
        Time series from which the extremes were extracted (default=None).
    figsize : tuple, optional
        Figure size in inches (default=(8, 5)).

    Returns
    -------
    figure : matplotlib.pyplot.Figure
        Figure object.
    axes : matplotlib.pyplot.Axes
        Axes object
    """
    # TODO - ask or infer block size, ask or infer threshold
    #   for block show thin dashed lines in the background
    #   for threshold show thin horizontal line
    #   remove grid
    with plt.rc_context(rc=pyextremes_rc):
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            extremes.index, extremes.values,
            s=20, lw=.5, edgecolor='w', facecolor='#F85C50', zorder=10
        )
        if ts is not None:
            ax.plot(
                ts.index, ts.values,
                ls='-', color='#5199FF', lw=0.25, zorder=5
            )
        ax.set_ylabel(extremes.name)
        ax.set_xlabel(extremes.index.name)

        return fig, ax


if __name__ == '__main__':
    pass
    # import os
    # import pathlib
    #
    # test_extremes = pd.read_csv(
    #     pathlib.Path(os.getcwd()) / r'tests/data/extremes_bm_high.csv',
    #     index_col=0, parse_dates=True, squeeze=True
    # )
    # test_ts = pd.read_csv(
    #     pathlib.Path(os.getcwd()) / r'tests/data/battery_wl.csv',
    #     index_col=0, parse_dates=True, squeeze=True
    # )
    # plot_extremes(extremes=test_extremes, ts=test_ts)
