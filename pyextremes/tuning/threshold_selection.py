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
import matplotlib.gridspec
import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.extremes import get_extremes, ExtremesTransformer
from pyextremes.plotting import pyextremes_rc
from pyextremes import EVA

logger = logging.getLogger(__name__)


def get_default_thresholds(
        ts: pd.Series,
        extremes_type: str,
        num: int = 100
) -> np.ndarray:
    if extremes_type == 'high':
        start = np.quantile(ts.values, 0.9)
        stop = ts.sort_values(ascending=False).iloc[9]
    elif extremes_type == 'low':
        start = np.quantile(ts.values, 0.1)
        stop = ts.sort_values(ascending=True).iloc[9]
    else:
        raise ValueError(f'\'{extremes_type}\' is not a valid value of the \'extremes_type\' argument')
    return np.linspace(start=start, stop=stop, num=100)


def plot_mean_residual_life(
        ts: pd.Series,
        thresholds: typing.Union[list, np.ndarray] = None,
        extremes_type: str = 'high',
        figsize: tuple = (8, 5)
) -> tuple:

    thresholds = None
    extremes_type = 'high'

    if thresholds is None:
        logger.info('calculating default threshold values')
        thresholds = get_default_thresholds(ts=ts, extremes_type=extremes_type)

    logger.info('calculating mean residual life for each threshold')
    mean_residual_lives = []
    mrl_confidence = []
    for threshold in thresholds:
        if extremes_type == 'high':
            exceedances = ts.loc[ts > threshold] - threshold
        elif extremes_type == 'low':
            exceedances = ts.loc[ts < threshold] - threshold
        else:
            raise ValueError(f'\'{extremes_type}\' is not a valid value of the \'extremes_type\' argument')
        mean_residual_lives.append(exceedances.mean())
        mrl_confidence.append(
            scipy.stats.norm.interval(
                alpha=0.95, loc=exceedances.mean(),
                scale=exceedances.std(ddof=1) / np.sqrt(len(exceedances))
            )
        )

    with plt.rc_context(rc=pyextremes_rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=96)

        logger.info('configuring axes')
        ax.grid(False)

        logger.info('plotting mean residual life')
        ax.plot(
            thresholds, mean_residual_lives,
            color='#F85C50', lw=2, ls='-', zorder=20
        )

        logger.info('plotting confidence intervals')
        for ci in np.transpose(mrl_confidence):
            ax.plot(
                thresholds, ci,
                color='#5199FF', lw=1, ls='--', zorder=15
            )
        ax.fill_between(
            thresholds, *np.transpose(mrl_confidence),
            facecolor='#5199FF', edgecolor='None', alpha=0.25, zorder=10
        )

        logger.info('labeling axes')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Mean excess')

        return fig, ax


def plot_parameter_stability(
        ts: pd.Series,
        thresholds: typing.Union[list, np.ndarray] = None,
        r: typing.Union[str, pd.Timedelta] = '24H',
        extremes_type: str = 'high',
        figsize: tuple = (8, 5)
) -> tuple:
    thresholds = None
    extremes_type = 'high'

    if thresholds is None:
        logger.info('calculating default threshold values')
        thresholds = get_default_thresholds(ts=ts, extremes_type=extremes_type)

    shape_parameters = []
    scale_parameters = []
    for threshold in thresholds:
        extremes = get_extremes(
            ts=ts,
            method='POT',
            extremes_type=extremes_type,
            threshold=threshold,
            r=r
        )
        extremes_transformer = ExtremesTransformer(extremes=extremes, extremes_type=extremes_type)
        c, loc, scale = scipy.stats.genpareto.fit(
            data=extremes_transformer.transformed_extremes,
            floc=threshold
        )
        shape_parameters.append(c)
        scale_parameters.append(scale - c * threshold)

    with plt.rc_context(rc=pyextremes_rc):
        logger.info('creating figure')
        fig = plt.figure(figsize=figsize, dpi=96)

        logger.info('creating gridspec')
        gs = matplotlib.gridspec.GridSpec(
            nrows=2,
            ncols=1,
            wspace=0.1,
            hspace=0.1,
            width_ratios=[1],
            height_ratios=[1, 1]
        )

        logger.info('creating and configuring axes')
        ax_shape = fig.add_subplot(gs[0, 0])
        ax_scale = fig.add_subplot(gs[1, 0])

        logger.info('plotting parameters')
        ax_shape.plot(thresholds, shape_parameters, ls='-', color='#1771F1', lw=2, zorder=5)
        ax_scale.plot(thresholds, scale_parameters, ls='-', color='#1771F1', lw=2, zorder=5)

        logger.info('configuring axes')
        ax_shape.tick_params(axis='x', which='both', labelbottom=False, length=0)
        if extremes_type == 'low':
            ax_shape.set_xlim(ax_shape.get_xlim()[::-1])
        ax_scale.set_xlim(ax_shape.get_xlim())

        logger.info('labeling axes')
        ax_shape.set_ylabel(r'Shape, $\xi$')
        ax_scale.set_ylabel(r'Modified scale, $\sigma^*$')
        ax_scale.set_xlabel('Threshold')

        return fig, (ax_shape, ax_scale)


def plot_return_value_stability(
        ts: pd.Series,
        return_period: typing.Union[int, float],
        return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
        thresholds: typing.Union[list, np.ndarray] = None,
        r: typing.Union[str, pd.Timedelta] = '24H',
        extremes_type: str = 'high',
        figsize: tuple = (8, 5)
) -> tuple:
    thresholds = None
    extremes_type = 'high'
    return_period = 100

    if thresholds is None:
        logger.info('calculating default threshold values')
        thresholds = get_default_thresholds(ts=ts, extremes_type=extremes_type)

    return_values = []
    model = EVA(data=ts)
    for threshold in thresholds:
        model.get_extremes(
            method='POT',
            extremes_type=extremes_type,
            threshold=threshold,
            r=r
        )
        model.fit_model()
        return_values.append(
            model.get_return_value(
                return_period=return_period,
                return_period_size=return_period_size,
                alpha=None
            )[0]
        )

    with plt.rc_context(rc=pyextremes_rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=96)

        logger.info('configuring axes')
        ax.grid(False)

        logger.info('plotting return values')
        ax.plot(
            thresholds, return_values,
            color='#1771F1', lw=2, ls='-', zorder=20
        )

        logger.info('labeling axes')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Return value')

        return fig, ax

if __name__ == '__main__':
    import os
    import pathlib
    test_path = pathlib.Path(os.getcwd()) / 'tests' / 'data' / 'battery_wl.csv'
    test_ts = (
        pd.read_csv(test_path, index_col=0, parse_dates=True, squeeze=True)
        .sort_index(ascending=True)
        .dropna()
    )
    test_ts = test_ts.loc[pd.to_datetime('1925'):]
    test_ts = test_ts - (test_ts.index.array - pd.to_datetime('1992')) / pd.to_timedelta('1Y') * 2.87e-3

