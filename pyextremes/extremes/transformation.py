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
    Provides methods to perform forward and inverse transforms of extreme values.
    Transformation of extreme value series allows using distributions previously unavailable
    due to support constraints.

    Transformation is performed by mirroring the extreme values against 0 (multiplication by -1)
    and then shifting the resulting series to have start at 0 (min or max, depending on original orientation).

    Parameters
    ----------
    extremes : pandas.Series
        Time series of extreme events.
    extremes_type : str
        high - provided extreme values are extreme high values
        low - provided extreme values are extreme low values
    null_transform : bool, optional
        If True, then the transformer doesn't perform any transformations.
    """

    def __init__(
            self,
            extremes: pd.Series,
            extremes_type: str,
            null_transform: bool = False
    ) -> None:

        if null_transform:
            logger.info('declaring a null transformer')
            self.pivot_point = None
        else:
            logger.info(f'preparing a transformer for \'{extremes_type}\' extremes')
            if extremes_type == 'high':
                self.pivot_point = extremes.min()
            elif extremes_type == 'low':
                self.pivot_point = extremes.max()
            else:
                raise ValueError(f'\'{extremes_type}\' is not a valid \'extremes_type\' value')

        logger.info('transforming extremes')
        self.transformed_extremes = self.transform(value=extremes)

    def transform(
            self,
            value: typing.Union[float, pd.Series]
    ) -> typing.Union[float, pd.Series]:
        """
        Perform a forward/inverse transform of extreme value(s).
        Forward and inverse transforms are identical.

        Parameters
        ----------
        value : float or pandas.Series
            Value or series of values to be transformed.

        Returns
        -------
        transformed_value : float or pandas.Series
            Transformed value or series.
        """

        logger.debug('performing transform')
        if self.pivot_point is None:
            return value
        else:
            return self.pivot_point - value
