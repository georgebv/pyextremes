import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import typing


theme_color = '#454545'
pyextremes_rc = {
    'font.family': 'arial',
    'font.size': 10,
    'text.color': theme_color,
    'axes.edgecolor': theme_color,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'axes.labelsize': 10,
    'axes.labelweight': 'normal',
    'axes.prop_cycle': matplotlib.cycler(
        'color',
        [
            '#1771F1',
            '#F85C50',
            '#35D073',
            '#FFC11E'
        ]
    ),
    'xtick.major.size': 2,
    'xtick.minor.size': 1,
    'xtick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'xtick.major.top': True,
    'xtick.major.bottom': True,
    'xtick.minor.top': True,
    'xtick.minor.bottom': True,
    'xtick.color': theme_color,
    'ytick.major.size': 2,
    'ytick.minor.size': 1,
    'ytick.major.width': 0.8,
    'ytick.minor.width': 0.6,
    'ytick.color': theme_color,
    'ytick.major.left': True,
    'ytick.major.right': True,
    'ytick.minor.left': True,
    'ytick.minor.right': True,
    'grid.color': theme_color,
    'grid.linestyle': ':',
    'grid.linewidth': 0.4,
    'grid.alpha': 1.0,
    'legend.frameon': False,
    'legend.edgecolor': theme_color,
    'figure.figsize': (8, 5),
    'figure.dpi': 96
}


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
