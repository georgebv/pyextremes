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

import pandas as pd

from pyextremes.extremes import get_extremes


class EVA:

    def __init__(
            self,
            data: pd.Series
    ):
        self.data = data
        self.extremes = None

    def get_extremes(
            self,
            method: str,
            extremes_type: str = 'high',
            **kwargs
    ):
        """
        Get extreme events from a signal time series using specified method.
        Stores extreme values in the self.extremes attribute.

        Parameters
        ----------
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
                    raise (default) - raise error for blocks with no data
                    ignore - ignore blocks with no data
                    coerce - get extreme values for blocks with no data as mean of all other extreme events
                        in the series with index being the middle point of corresponding interval
            if method is POT:
                threshold : int or float
                    Threshold used to find exceedances.
                r : str or pandas.Timedelta, optional
                    Duration of window used to decluster the exceedances (default='24H').
        """

        self.extremes = get_extremes(method=method, ts=self.data, extremes_type=extremes_type, **kwargs)
