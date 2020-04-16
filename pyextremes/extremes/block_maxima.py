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

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_extremes_block_maxima(
        ts: pd.Series,
        extremes_type: str,
        block_size: typing.Union[str, pd.Timedelta],
        errors: str
) -> pd.Series:
    """
    Get extreme events from a signal time series using the Block Maxima method.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes_type : str
        high - get extreme high values
        low - get extreme low values
    block_size : str or pandas.Timedelta
        Block size.
    errors : str
        raise - raise error for blocks with no data
        ignore - ignore blocks with no data
        coerce - get extreme values for blocks with no data as mean of all other extreme events
            in the series with index being the middle point of corresponding interval

    Returns
    -------
    extremes : pandas.Series
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
        index=pd.Index(data=extreme_indices, name=ts.index.name or 'date-time'),
        name=ts.name
    ).fillna(np.nanmean(extreme_values))
