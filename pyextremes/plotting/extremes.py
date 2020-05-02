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
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyextremes.plotting.style import pyextremes_rc

logger = logging.getLogger(__name__)


def plot_extremes(
        ts: pd.Series,
        extremes: pd.Series,
        extremes_method: str,
        extremes_type: str,
        block_size: typing.Union[str, pd.Timedelta] = None,
        figsize: tuple = (8, 5)
):
    """
    Plot time series of extreme events.

    Parameters
    ----------
    ts : pandas.Series
        Time series from which the extremes were extracted.
    extremes : pandas.Series
        Time series of extreme events.
    extremes_method : str
        Extreme value extraction method.
        Supported values: BM or POT.
    extremes_type : str
        high - get extreme high values
        low - get extreme low values
    block_size : str or pandas.Timedelta, optional
        Block size in the 'BM' extremes_method (default=None).
        If None, then is calculated as median distance between extreme events.
    figsize : tuple, optional
        Figure size in inches (default=(8, 5)).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object.
    axes : matplotlib.axes.Axes
        Axes object.
    """

    if extremes_method == 'BM':
        logger.info('parsing block_size')
        if block_size is None:
            logger.info('calculating block_size as mean of distances between extremes')
            block_size = pd.to_timedelta(np.quantile(np.diff(extremes.index), 0.5))
        else:
            if not isinstance(block_size, pd.Timedelta):
                if isinstance(block_size, str):
                    logger.info('converting block_size to timedelta')
                    block_size = pd.to_timedelta(block_size)
                else:
                    raise TypeError(f'invalid type in {type(block_size)} for the \'block_size\' argument')
    else:
        if block_size is not None:
            raise ValueError('\'block_size\' value is applicable only if \'extremes_method\' is \'BM\'')

    with plt.rc_context(rc=pyextremes_rc):
        logger.info('creating figure')
        fig, ax = plt.subplots(figsize=figsize, dpi=96)
        ax.grid(False)

        logger.info('plotting extreme events')
        ax.scatter(
            extremes.index, extremes.values,
            s=20, lw=.5, edgecolor='w', facecolor='#F85C50', zorder=15
        )

        logger.info('plotting signal time series')
        ax.plot(
            ts.index, ts.values,
            ls='-', color='#5199FF', lw=0.25, zorder=10
        )

        logger.info('labelling the axes')
        ax.set_ylabel(extremes.name)
        ax.set_xlabel(extremes.index.name)

        if extremes_method == 'BM':
            logger.info('plotting block boundaries')
            start = ts.index[0]
            while start < ts.index[-1] + block_size:
                ax.axvline(
                    start, ls='--', lw=0.5, color='#D1D3D4', zorder=5
                )
                start += block_size
        elif extremes_method == 'POT':
            logger.info('plotting threshold line')
            if extremes_type == 'high':
                y = extremes.min()
            elif extremes_type == 'low':
                y = extremes.max()
            else:
                raise ValueError(f'\'{extremes_type}\' is not a valid \'extremes_type\' value')
            ax.axhline(y, ls='--', lw=1, color='#FF756B', zorder=5)
        else:
            raise ValueError(f'\'{extremes_method}\' is not a valid \'extremes_method\' value')

        return fig, ax
