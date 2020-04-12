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

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_extremes(
        method: str,
        **kwargs
) -> pd.Series:
    logger.debug(f'calling get_extremes with method={method}')
    if method == 'BM':
        return _get_extremes_bm(**kwargs)
    elif method == 'POT':
        return _get_extremes_pot(**kwargs)
    else:
        raise ValueError(f'{method} is not a valid method value')


def _get_extremes_bm(
        ts: pd.Series,
        block_size: pd.Timedelta,
        extremes_type: str,
        errors: str
) -> pd.Series:
    """
    Get extreme events from a signal time series using the Block Maxima method.

    Parameters
    ----------
    ts : pd.Series
        Time series of the signal.
    block_size : pd.Timedelta
        Block size.
    extremes_type : str
        high - get highest value within each block
        low - get lowest value within each block
    errors : str
        coerce - get extreme values for blocks with no data as mean of all other extreme events in the series
            with index being the middle point of corresponding interval
        ignore - ignore blocks with no data
        raise - raise error for blocks with no data

    Returns
    -------
    extremes : pd.Series
        Time series of extreme events.
    """

    logger.debug(f'getting extreme extraction function for extremes_type={extremes_type}')
    if extremes_type == 'high':
        extremes_func = pd.Series.idxmax
    elif extremes_type == 'low':
        extremes_func = pd.Series.idxmin
    else:
        raise ValueError(f'{extremes_type} is not a valid extremes_type value')

    logger.debug('preparing date_time_intervals')
    periods = (ts.index[-1] - ts.index[0]) / block_size
    if periods % 1 == 0:
        periods = int(periods)
    else:
        periods = int(periods) + 1
    date_time_intervals = pd.interval_range(start=ts.index[0], freq=block_size, periods=periods, closed='left')

    logger.debug('collecting extreme events')
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
                raise ValueError(f'no data in block [{interval.left} ; {interval.right})')
            else:
                raise ValueError(f'{errors} is not a valid errors value')

    if empty_intervals > 0:
        logger.warning(f'{empty_intervals} blocks contained no data')

    logger.debug('successfully collected extreme events, returning the series')
    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name='date-time'),
        name=ts.name
    ).fillna(np.nanmean(extreme_values))


def _get_extremes_pot(
        ts: pd.Series
) -> pd.Series:
    raise NotImplementedError


def get_return_period(
        extremes: pd.Series
) -> list:
    raise NotImplementedError


if __name__ == '__main__':
    pass
    # import os
    # import pathlib

    # ts = pd.read_csv(
    #     pathlib.Path(os.getcwd()) / r'tests/data/battery_wl.csv',
    #     index_col=0, parse_dates=True, squeeze=True
    # )
    # block_size = pd.to_timedelta('365.2425D')
    # extremes_type = 'high'
    # errors = 'coerce'
