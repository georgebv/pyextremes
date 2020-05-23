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

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)

plotting_positions = {
    'ecdf': (0, 1),
    'hazen': (0.5, 0.5),
    'weibull': (0, 0),
    'tukey': (1 / 3, 1 / 3),
    'blom': (3 / 8, 3 / 8),
    'median': (0.3175, 0.3175),
    'cunnane': (0.4, 0.4),
    'gringorten': (0.44, 0.44),
    'beard': (0.31, 0.31)
}


def get_return_periods(
        ts: pd.Series,
        extremes: pd.Series,
        extremes_method: str,
        extremes_type: str,
        block_size: typing.Union[str, pd.Timedelta] = None,
        return_period_size: typing.Union[str, pd.Timedelta] = '1Y',
        plotting_position: str = 'weibull'
) -> pd.DataFrame:
    """
    Calculate return periods for given extreme values and plotting position.
    Return periods have units of return_period_size.
    Plotting positions were taken from https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes : pandas.Series
        Time series of extreme events.
    extremes_method : str
        Extreme value extraction method.
        Supported values: BM or POT.
    extremes_type : str
        high - provided extreme values are extreme high values
        low - provided extreme values are extreme low values
    block_size : str or pandas.Timedelta, optional
        Block size in the 'BM' extremes_method (default=None).
        If None, then is calculated as median distance between extreme events.
    return_period_size : str or pandas.Timedelta, optional
        Size of return periods (default='1Y').
        If set to '30D', then a return period of 12 would be roughly equivalent to 1 year return period.
    plotting_position : str, optional
        Plotting position name (default='weibull'), not case-sensitive.
        Supported plotting positions:
            ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard

    Returns
    -------
    extreme_events : pandas.DataFrame
        A DataFrame with extreme values, exceedance probabilities, and return periods in units of return_period_size.
    """

    if extremes_method == 'BM':
        logger.info('parsing the \'block_size\' argument')
        if block_size is None:
            logger.info('calculating \'block_size\' as median distance between extremes')
            block_size = pd.to_timedelta(np.quantile(np.diff(extremes.index), 0.5))
        else:
            if not isinstance(block_size, pd.Timedelta):
                if isinstance(block_size, str):
                    logger.info('converting \'block_size\' to pandas.Timedelta')
                    block_size = pd.to_timedelta(block_size)
                else:
                    raise TypeError(f'invalid type in {type(block_size)} for the \'block_size\' argument')
    else:
        if block_size is not None:
            raise ValueError('\'block_size\' value is applicable only if \'extremes_method\' is \'BM\'')

    logger.info('parsing the \'return_period_size\' argument')
    if not isinstance(return_period_size, pd.Timedelta):
        if isinstance(return_period_size, str):
            logger.info('converting \'return_period_size\' to pandas.Timedelta')
            return_period_size = pd.to_timedelta(return_period_size)
        else:
            raise TypeError(f'invalid type in {type(return_period_size)} for the \'return_period_size\' argument')

    logger.info('calculating rate of extreme events as number of events per one return period')
    if extremes_method == 'BM':
        logger.debug('calculating \'extremes_rate\' for BM method')
        extremes_rate = return_period_size / block_size
    elif extremes_method == 'POT':
        logger.debug('calculating \'extremes_rate\' for POT method')
        n_periods = (ts.index.max() - ts.index.min()) / return_period_size
        extremes_rate = len(extremes) / n_periods
    else:
        raise ValueError(f'\'{extremes_method}\' is not a valid value for the \'extremes_method\' argument')

    logger.info('ranking the extreme values from most extreme (1) to least extreme (len(extremes))')
    if extremes_type == 'high':
        ranks = len(extremes) + 1 - scipy.stats.rankdata(extremes.values, method='average')
    elif extremes_type == 'low':
        ranks = scipy.stats.rankdata(extremes.values, method='average')
    else:
        raise ValueError(f'\'{extremes_type}\' is not a valid value for the \'extremes_type\' argument')

    logger.info('getting plotting position parameters')
    try:
        alpha, beta = plotting_positions[plotting_position.lower()]
    except KeyError:
        raise ValueError(f'\'{plotting_position}\' is not a valid value for the \'plotting_position\' argument')

    logger.info('caclucating exceedance probabilities')
    exceedance_probability = (ranks - alpha) / (len(extremes) + 1 - alpha - beta)

    logger.info('calculating return periods')
    return_periods = 1 / exceedance_probability / extremes_rate

    logger.info('successfully calculated return periods, returning the DataFrame')
    return pd.DataFrame(
        data={
            extremes.name: extremes.values,
            'exceedance probability': exceedance_probability,
            'return period': return_periods
        },
        index=extremes.index
    )
