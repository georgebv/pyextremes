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

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_extremes(
        method: str,
        ts: pd.Series,
        extremes_type: str = 'high',
        **kwargs
) -> pd.Series:
    """
    Get extreme events from a signal time series using specified method.

    Parameters
    ----------
    method : str
        Extreme value extraction method.
        Supported values: BM or POT.
    ts : pd.Series
        Time series of the signal.
    extremes_type : str, optional
        high (default) - get extreme high values
        low - get extreme low values
    kwargs
        if method is BM:
            block_size : str or pd.Timedelta, optional
                Block size (default='1Y').
            errors : str, optional
                raise (default) - raise error for blocks with no data
                ignore - ignore blocks with no data
                coerce - get extreme values for blocks with no data as mean of all other extreme events
                    in the series with index being the middle point of corresponding interval
        if method is POT:
            threshold : int or float
                Threshold used to find exceedances.
            r : str or pd.Timedelta, optional
                Duration of window used to decluster the exceedances (default='24H').

    Returns
    -------
    extremes : pd.Series
        Time series of extreme events.
    """

    logger.debug(f'calling get_extremes with method={method}')
    if method == 'BM':
        block_size = kwargs.pop('block_size', '1Y')
        errors = kwargs.pop('errors', 'raise')
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
        return _get_extremes_bm(ts=ts, extremes_type=extremes_type, block_size=block_size, errors=errors)
    elif method == 'POT':
        threshold = kwargs.pop('threshold')
        r = kwargs.pop('r', '24H')
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
        return _get_extremes_pot(ts=ts, extremes_type=extremes_type, threshold=threshold, r=r)
    else:
        raise ValueError(f'{method} is not a valid method value')


def _get_extremes_bm(
        ts: pd.Series,
        extremes_type: str,
        block_size: typing.Union[str, pd.Timedelta],
        errors: str
) -> pd.Series:
    """
    Get extreme events from a signal time series using the Block Maxima method.

    Parameters
    ----------
    ts : pd.Series
        Time series of the signal.
    extremes_type : str
        high - get extreme high values
        low - get extreme low values
    block_size : str or pd.Timedelta
        Block size.
    errors : str
        raise - raise error for blocks with no data
        ignore - ignore blocks with no data
        coerce - get extreme values for blocks with no data as mean of all other extreme events
            in the series with index being the middle point of corresponding interval

    Returns
    -------
    extremes : pd.Series
        Time series of extreme events.
    """

    logger.info(f'getting extreme extraction function for extremes_type={extremes_type}')
    if extremes_type == 'high':
        extremes_func = pd.Series.idxmax
    elif extremes_type == 'low':
        extremes_func = pd.Series.idxmin
    else:
        raise ValueError(f'{extremes_type} is not a valid extremes_type value')

    logger.info('parsing block_size')
    if not isinstance(block_size, pd.Timedelta):
        if isinstance(block_size, str):
            block_size = pd.to_timedelta(block_size)
        else:
            raise TypeError(f'invalid type in {type(block_size)} for the block_size argument')
    if block_size <= pd.to_timedelta('1D'):
        logger.warning(f'block_size {block_size} is too small, consider changing')

    logger.info('preparing date_time_intervals')
    periods = int(np.ceil((ts.index[-1] - ts.index[0]) / block_size))
    date_time_intervals = pd.interval_range(start=ts.index[0], freq=block_size, periods=periods, closed='left')

    logger.info('collecting extreme events')
    empty_intervals = 0
    extreme_indices, extreme_values = [], []
    for i, interval in enumerate(date_time_intervals):
        logger.debug(f'processing block [{interval.left} ; {interval.right})')
        ts_slice = ts.loc[
            (ts.index >= interval.left) &
            (ts.index < interval.right)
        ].dropna()
        if len(ts_slice) > 0:
            extreme_indices.append(extremes_func(ts_slice, skipna=True))
            extreme_values.append(ts_slice.loc[extreme_indices[-1]])
        else:
            empty_intervals += 1
            if errors == 'coerce':
                logger.debug(f'coerced error in block [{interval.left} ; {interval.right})')
                extreme_indices.append(interval.mid)
                extreme_values.append(np.nan)
            elif errors == 'ignore':
                logger.debug(f'ignored error in block [{interval.left} ; {interval.right})')
            elif errors == 'raise':
                raise ValueError(
                    f'no data in block [{interval.left} ; {interval.right}), fille the gaps in the data '
                    f'or set the argument \'errors\' to \'coerce\' or \'ignore\''
                )
            else:
                raise ValueError(f'{errors} is not a valid errors value')

    if empty_intervals > 0:
        logger.warning(f'{empty_intervals} blocks contained no data')

    logger.info('successfully collected extreme events, returning the series')
    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name='date-time'),
        name=ts.name
    ).fillna(np.nanmean(extreme_values))


def _get_extremes_pot(
        ts: pd.Series,
        extremes_type: str,
        threshold: typing.Union[int, float],
        r: typing.Union[str, pd.Timedelta]
) -> pd.Series:
    """
    Get extreme events from a signal time series using the Peaks Over Threshold method.

    Parameters
    ----------
    ts : pd.Series
        Time series of the signal.
    extremes_type : str
        high - get extreme high values
        low - get extreme low values
    threshold : int or float
        Threshold used to find exceedances.
    r : str or pd.Timedelta
        Duration of window used to decluster the exceedances.

    Returns
    -------
    extremes : pd.Series
        Time series of extreme events.
    """

    logger.info(f'collecting exceedances for extremes_type={extremes_type}')
    if extremes_type == 'high':
        exceedances = ts.loc[ts.values > threshold].dropna()
        comparison_function = np.greater
    elif extremes_type == 'low':
        exceedances = ts.loc[ts.values < threshold].dropna()
        comparison_function = np.less
    else:
        raise ValueError(f'{extremes_type} is not a valid extremes_type value')

    logger.info('parsing r')
    if not isinstance(r, pd.Timedelta):
        if isinstance(r, str):
            r = pd.to_timedelta(r)
        else:
            raise TypeError(f'invalid type in {type(r)} for the r argument')

    logger.info('declustering exceedances')
    extreme_indices, extreme_values = [exceedances.index[0]], [exceedances.values[0]]
    for index, value in exceedances.iteritems():
        if (index - extreme_indices[-1]) > r:
            logger.debug(f'started a new cluster with {index} {value}')
            extreme_indices.append(index)
            extreme_values.append(value)
        else:
            if comparison_function(value, extreme_values[-1]):
                logger.debug(f'found a new cluster peak in {index} {value}')
                extreme_indices[-1] = index
                extreme_values[-1] = value

    logger.info('successfully collected extreme events, returning the series')
    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name='date-time'),
        name=ts.name
    )


def get_return_periods(
        ts: pd.Series,
        extremes: pd.Series,
        extremes_method: str,
        extremes_type: str,
        plotting_position: str = 'weibull',
) -> pd.DataFrame:
    """
    Calculate return periods in years for given extreme values and plotting position.
    Plotting positions taken from https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html

    Parameters
    ----------
    ts : pd.Series
        Time series of the signal.
    extremes : pd.Series
        Time series of extreme events.
    extremes_method : str
        Extreme value extraction method.
        Supported values: BM or POT.
    extremes_type : str
        high - get extreme high values
        low - get extreme low values
    plotting_position : str, optional
        Plotting position name (default='weibull'), not case-sensitive.
        Supported plotting positions:
            ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard

    Returns
    -------
    extreme_events : pd.DataFrame
        A DataFrame with extreme values and corresponding return periods.
    """

    logger.info('calculating rate of extreme events [events per year]')
    if extremes_method == 'BM':
        extremes_rate = 1
        logger.debug('calculated rate for BM method')
    elif extremes_method == 'POT':
        n_years = (ts.index[-1] - ts.index[0]) / pd.to_timedelta('1Y')
        extremes_rate = len(extremes) / n_years
        logger.debug('calculated rate for POT method')
    else:
        raise ValueError(f'{extremes_method} is not a valid extremes_method value')

    logger.info('ranking the extreme values')
    if extremes_type == 'high':
        ranks = len(extremes) + 1 - scipy.stats.rankdata(extremes.values, method='average')
    elif extremes_type == 'low':
        ranks = scipy.stats.rankdata(extremes.values, method='average')
    else:
        raise ValueError(f'{extremes_type} is not a valid extremes_type value')

    logger.info('getting plotting position parameters')
    plotting_positions = {
        'ecdf': (0, 1),
        'hazen': (0.5, 0.5),
        'weibull': (0, 0),
        'tukey': (1/3, 1/3),
        'blom': (3/8, 3/8),
        'median': (0.3175, 0.3175),
        'cunnane': (0.4, 0.4),
        'gringorten': (0.44, 0.44),
        'beard': (0.31, 0.31)
    }
    try:
        alpha = plotting_positions[plotting_position.lower()][0]
        beta = plotting_positions[plotting_position.lower()][1]
    except KeyError:
        raise ValueError(f'{plotting_position} is not a valid plotting_position value')

    logger.info('caclucating exceedance probabilities')
    exceedance_probability = (ranks - alpha) / (len(extremes) + 1 - alpha - beta)

    logger.info('successfully calculated return periods, returning the DataFrame')
    return pd.DataFrame(
        data={
            extremes.name: extremes.values,
            'return period [yr]': 1 / exceedance_probability / extremes_rate
        },
        index=extremes.index
    )


if __name__ == '__main__':
    pass
    # import os
    # import pathlib
    #
    # ts = pd.read_csv(
    #     pathlib.Path(os.getcwd()) / r'tests/data/battery_wl.csv',
    #     index_col=0, parse_dates=True, squeeze=True
    # )
    # block_size = pd.to_timedelta('365.2425D')
    # extremes_type = 'high'
    # errors = 'coerce'
    #
    # threshold = 1
    # r = '24H'
    #
    # method = 'POT'
    # extremes_type = 'high'
    # extremes = get_extremes(
    #     ts=ts,
    #     method=method,
    #     threshold=1.35,
    #     r='24H',
    #     extremes_type=extremes_type
    # )
