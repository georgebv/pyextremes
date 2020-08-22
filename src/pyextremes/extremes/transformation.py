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

import pandas as pd

logger = logging.getLogger(__name__)


class ExtremesTransformer:
    """
    Extreme value transformer class.
    Provides methods to transform extreme value series to and from format compatible with pyextremes models.
    The values are transformed in a way to have extreme events at the right side of the distribution support.
    For extremes_type being 'high' no transform is performed.
    For extremes_type being 'low' the values are flipped around the maximum value.

    Parameters
    ----------
    extremes : pandas.Series
        Time series of extreme events.
    extremes_type : str
        high - provided extreme values are extreme high values
        low - provided extreme values are extreme low values
    """

    def __init__(
            self,
            extremes: pd.Series,
            extremes_type: str,
    ) -> None:
        self.extremes = extremes
        self.extremes_type = extremes_type

        logger.info('transforming extremes')
        self.transformed_extremes = self.transform(value=extremes)

    def transform(
            self,
            value: typing.Union[None, float, pd.Series]
    ) -> typing.Union[None, float, pd.Series]:
        """
        Perform a transform of extreme values. Works both ways.

        Parameters
        ----------
        value : float or pandas.Series
            Value or series of values to be transformed.

        Returns
        -------
        transformed_value : None, float, or pandas.Series
            Transformed value or series.
            If value is None, returns None.
        """

        if value is None:
            return value
        else:
            logger.debug('performing transform')
            if self.extremes_type == 'high':
                return value
            else:
                return 2 * self.extremes.max() - value
