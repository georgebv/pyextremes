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

logger = logging.getLogger(__name__)


def get_extremes_peaks_over_threshold(
        ts: pd.Series,
        extremes_type: str,
        threshold: typing.Union[int, float],
        r: typing.Union[str, pd.Timedelta] = '24H'
) -> pd.Series:
    """
    Get extreme events from a signal time series using the Peaks Over Threshold method.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes_type : str
        high - get extreme high values
        low - get extreme low values
    threshold : int or float
        Threshold used to find exceedances.
    r : str or pandas.Timedelta, optional
        Duration of window used to decluster the exceedances (default='24H').

    Returns
    -------
    extremes : pandas.Series
        Time series of extreme events.
    """

    logger.info(f'getting extreme value extraction function for extremes_type={extremes_type}')
    if extremes_type == 'high':
        comparison_function = np.greater
    elif extremes_type == 'low':
        comparison_function = np.less
    else:
        raise ValueError(f'\'{extremes_type}\' is not a valid value of the \'extremes_type\' argument')

    logger.info('getting exceedances')
    exceedances = ts.loc[comparison_function(ts.values, threshold)]

    logger.info('parsing the \'r\' argument')
    if not isinstance(r, pd.Timedelta):
        if isinstance(r, str):
            logger.info('converting \'r\' to pandas.Timedelta')
            r = pd.to_timedelta(r)
        else:
            raise TypeError(f'invalid type in {type(r)} for the \'r\' argument')

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
        index=pd.Index(data=extreme_indices, name=ts.index.name or 'date-time'),
        name=ts.name
    )
