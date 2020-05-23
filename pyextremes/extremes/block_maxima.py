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
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NoDataBlockWarning(Warning):
    pass


def get_extremes_block_maxima(
        ts: pd.Series,
        extremes_type: str,
        block_size: typing.Union[str, pd.Timedelta] = '1Y',
        errors: str = 'raise'
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
    block_size : str or pandas.Timedelta, optional
        Block size (default='1Y').
    errors : str, optional
        raise (default) - raise an exception when encountering a block with no data
        ignore - ignore blocks with no data
        coerce - get extreme values for blocks with no data as mean of all other extreme events
            in the series with index being the middle point of corresponding interval

    Returns
    -------
    extremes : pandas.Series
        Time series of extreme events.
    """

    logger.info(f'getting extreme value extraction function for extremes_type={extremes_type}')
    if extremes_type == 'high':
        extremes_func = pd.Series.idxmax
    elif extremes_type == 'low':
        extremes_func = pd.Series.idxmin
    else:
        raise ValueError(f'\'{extremes_type}\' is not a valid value of the \'extremes_type\' argument')

    logger.info('parsing the \'block_size\' argument')
    if not isinstance(block_size, pd.Timedelta):
        if isinstance(block_size, str):
            logger.info('converting \'block_size\' to pandas.Timedelta')
            block_size = pd.to_timedelta(block_size)
        else:
            raise TypeError(f'invalid type in {type(block_size)} for the \'block_size\' argument')

    logger.info('preparing date-time intervals')
    periods = int(np.ceil((ts.index.max() - ts.index.min()) / block_size))
    date_time_intervals = pd.interval_range(start=ts.index[0], freq=block_size, periods=periods, closed='left')

    logger.info('collecting extreme events')
    empty_intervals = 0
    extreme_indices, extreme_values = [], []
    for interval in date_time_intervals:
        logger.debug(f'processing block [{interval.left} ; {interval.right})')
        ts_slice = ts.loc[
            (ts.index >= interval.left) &
            (ts.index < interval.right)
        ]
        if len(ts_slice) > 0:
            extreme_indices.append(extremes_func(ts_slice))
            extreme_values.append(ts_slice.loc[extreme_indices[-1]])
        else:
            empty_intervals += 1
            if errors == 'coerce':
                logger.debug(f'coerced an error in block [{interval.left} ; {interval.right})')
                extreme_indices.append(interval.mid)
                extreme_values.append(np.nan)
            elif errors == 'ignore':
                logger.debug(f'ignored an error in block [{interval.left} ; {interval.right})')
            elif errors == 'raise':
                raise ValueError(
                    f'no data in block [{interval.left} ; {interval.right}), fill gaps in the data '
                    f'or set the argument \'errors\' to \'coerce\' or \'ignore\''
                )
            else:
                raise ValueError(f'\'{errors}\' is not a valid value of the \'errors\' argument')

    if empty_intervals > 0:
        warnings.warn(f'{empty_intervals} blocks contained no data', category=NoDataBlockWarning)

    logger.info('successfully collected extreme events, returning the series')
    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name=ts.index.name or 'date-time'),
        name=ts.name
    ).fillna(np.nanmean(extreme_values))
