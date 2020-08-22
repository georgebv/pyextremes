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

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from pyextremes.plotting.style import pyextremes_rc

logger = logging.getLogger(__name__)


def plot_probability(
        observed: np.ndarray,
        theoretical: np.ndarray,
        ax=None,
        figsize: tuple = (8, 8)
) -> tuple:
    """
    Plot a probability plot (QQ or PP).
    Used to assess goodness-of-fit of the model.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed values.
    theoretical : numpy.ndarray
        Theoretical values.
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
            logger.info('creating new Figure and Axes')
            fig, ax = plt.subplots(figsize=figsize, dpi=96)
        else:
            logger.info('plotting on existing Axes object')
            fig = None

        logger.info('configuring axes')
        ax.grid(False)

        logger.info('plotting scatter of observed and theoretical probabilities')
        ax.scatter(
            theoretical, observed,
            marker='o', s=20, lw=0.75, facecolor='k', edgecolor='w', zorder=10
        )

        logger.info('plotting diagonal perfect-fit line')
        min_value = min([min(ax.get_xlim()), min(ax.get_ylim())])
        max_value = max([max(ax.get_xlim()), max(ax.get_ylim())])
        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color='#5199FF', lw=1, ls='--', zorder=5
        )
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)

        logger.info('labeling axes')
        ax.set_xlabel('Theoretical')
        ax.set_ylabel('Observed')

        logger.info('calculating pearson R statistic and adding text to the figure')
        pearsonr, p_value = scipy.stats.pearsonr(theoretical, observed)
        axes_range = max_value - min_value
        ax.text(
            x=min_value + 0.05 * axes_range,
            y=max_value - 0.05 * axes_range,
            s=f'$R^2={pearsonr:.3f}$\n$p={p_value:.3f}$',
            horizontalalignment='left',
            verticalalignment='top'
        )

        return fig, ax
