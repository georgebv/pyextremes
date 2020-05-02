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

import pandas as pd

from pyextremes.extremes.block_maxima import get_extremes_block_maxima
from pyextremes.extremes.peaks_over_threshold import get_extremes_peaks_over_threshold

logger = logging.getLogger(__name__)

methods = {
    'BM': get_extremes_block_maxima,
    'POT': get_extremes_peaks_over_threshold
}


def get_extremes(
        ts: pd.Series,
        method: str,
        extremes_type: str = 'high',
        **kwargs
) -> pd.Series:
    """
    Get extreme events from a signal time series using a specified extreme value extraction method.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    method : str
        Extreme value extraction method.
        Supported values: BM or POT.
    extremes_type : str, optional
        high (default) - get extreme high values
        low - get extreme low values
    kwargs
        if method is BM:
            block_size : str or pandas.Timedelta, optional
                Block size (default='1Y').
            errors : str, optional
                raise (default) - raise an exception when encountering a block with no data
                ignore - ignore blocks with no data
                coerce - get extreme values for blocks with no data as mean of all other extreme events
                    in the series with index being the middle point of corresponding interval
        if method is POT:
            threshold : int or float
                Threshold used to find exceedances.
            r : str or pandas.Timedelta, optional
                Duration of window used to decluster the exceedances (default='24H').

    Returns
    -------
    extremes : pandas.Series
        Time series of extreme events.
    """

    logger.info(f'fetching \'{method}\' extreme value function')
    try:
        extremes_function = methods[method]
    except KeyError:
        raise ValueError(
            f'\'{method}\' is not a valid \'method\' value\n'
            f'Available methods: {", ".join(methods.keys())}'
        )
    logger.info('getting extreme values')
    return extremes_function(ts=ts, extremes_type=extremes_type, **kwargs)
