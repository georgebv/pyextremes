# pyextremes, Extreme Value Analysis in Python
# Copyright (C), 2020 Georgii Bocharov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import logging

import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from pyextremes.plotting.style import pyextremes_rc

logger = logging.getLogger(__name__)


def plot_return_values(
        observed_return_values: pd.DataFrame,
        modeled_return_values: pd.DataFrame,
        ax=None,
        figsize: tuple = (8, 8/1.618)
) -> tuple:
    """
    Plot return values and confidence intervals for given return periods.

    Parameters
    ----------
    observed_return_values : pandas.DataFrame
        DataFrame with observed return values.
        First column has extreme values.
        Column 'return periods' has return periods.
    modeled_return_values : pandas.DataFrame
        DataFrame with modeled return values.
        Index has return periods.
        Has these columns: 'return value', 'lower ci', 'upper ci'.
    ax : matplotlib.axes.Axes, optional
        Axes onto which the figure is drawn (default=None).
        If None, a new figure and axes is created.
    figsize : tuple, optional
        Figure size in inches (default=(8, 8/1.618)).

    Returns
    -------
    if ax is None:
        figure : matplotlib.figure.Figure
            Figure object.
    else:
        None
    axes : matplotlib.axes.Axes
        Axes object.
    """

    with plt.rc_context(rc=pyextremes_rc):
        if ax is None:
            logger.info('creating new figure and axes')
            fig, ax = plt.subplots(figsize=figsize, dpi=96)
        else:
            logger.info('plotting to existing axes')
            fig = None

        logger.info('configuring axes')
        ax.semilogx()
        ax.grid(True, which='both')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))

        logger.info('plotting return values')
        ax.plot(
            modeled_return_values.index,
            modeled_return_values.loc[:, 'return value'],
            color='#F85C50', lw=2, ls='-', zorder=20
        )

        logger.info('plotting confidence intervals')
        for col in ['lower ci', 'upper ci']:
            ax.plot(
                modeled_return_values.index,
                modeled_return_values.loc[:, col],
                color='#5199FF', lw=1, ls='--', zorder=15
            )
        ax.fill_between(
            modeled_return_values.index,
            modeled_return_values.loc[:, 'lower ci'],
            modeled_return_values.loc[:, 'upper ci'],
            facecolor='#5199FF', edgecolor='None', alpha=0.25, zorder=10
        )

        logger.info('plotting observed extreme values')
        ax.scatter(
            observed_return_values.loc[:, 'return period'],
            observed_return_values.loc[:, observed_return_values.columns[0]],
            marker='o', s=20, lw=1, facecolor='k', edgecolor='w', zorder=20
        )

        logger.info('labeling axes')
        ax.set_xlabel('Return period')
        ax.set_ylabel(observed_return_values.columns[0])

        return fig, ax
