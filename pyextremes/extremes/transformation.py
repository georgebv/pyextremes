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

    def __init__(
            self,
            extremes: pd.Series,
            extremes_method: str,
            extremes_type: str,
    ) -> None:
        """
        Extreme value transformer class. Provides methods to transform extreme value series to and from
        format compatible with pyextremes models.
        The transformed extreme value series have the following properties:
            - probability density decreases with increased value (same as if 'extremes_type' is 'high')
            - if 'extremes_method' is POT, then the series also starts at 0 (ensures that location parameter
                in the genpareto distribution is always 0)

        Parameters
        ----------
        extremes : pandas.Series
            Time series of extreme events.
        extremes_method : str
            Extreme value extraction method.
            Supported values: BM or POT.
        extremes_type : str
            high - provided extreme values are extreme high values
            low - provided extreme values are extreme low values
        """

        if extremes_method not in ['BM', 'POT']:
            raise ValueError(f'\'{extremes_method}\' is not a valid \'extremes_method\' value')
        if extremes_type not in ['high', 'low']:
            raise ValueError(f'\'{extremes_type}\' is not a valid \'extremes_type\' value')

        self.extremes_type = extremes_type

        logger.info('finding pivot value')
        if extremes_method == 'BM':
            self.pivot_value = 0
        elif extremes_method == 'POT':
            if extremes_type == 'high':
                self.pivot_value = extremes.min()
            elif extremes_type == 'low':
                self.pivot_value = extremes.max()
        logger.info(f'calculated the pivot value as {self.pivot_value}')

        logger.info('transforming extremes')
        self.transformed_extremes = self.forward_transform(value=extremes)

    def forward_transform(
            self,
            value: typing.Union[float, pd.Series]
    ) -> typing.Union[float, pd.Series]:
        """
        Perform an forward transform of extreme values.
        Transforms original extremes to pyextremes-model-friendly extremes.

        Parameters
        ----------
        value : float or pandas.Series
            Value or series of values to be transformed.

        Returns
        -------
        transformed_value : float or pandas.Series
            Transformed value or series.
        """

        logger.info('performing forward transform')
        if self.extremes_type == 'high':
            return value - self.pivot_value
        elif self.extremes_type == 'low':
            return self.__pivot(value=value) - self.pivot_value

    def inverse_transform(
            self,
            value: typing.Union[float, pd.Series]
    ) -> typing.Union[float, pd.Series]:
        """
        Perform an inverse transform of extreme values.
        Transforms pyextremes-model-friendly extremes to original extremes.

        Parameters
        ----------
        value : float or pandas.Series
            Value or series of values to be transformed.

        Returns
        -------
        transformed_value : float or pandas.Series
            Transformed value or series.
        """

        logger.info('performing inverse transform')
        if self.extremes_type == 'high':
            return value + self.pivot_value
        elif self.extremes_type == 'low':
            return self.__pivot(value=value+self.pivot_value)

    def __pivot(
            self,
            value: typing.Union[float, pd.Series]
    ) -> typing.Union[float, pd.Series]:
        """
        Pivot (a.k.a. mirror or rotate) series around self.pivot_value.

        Parameters
        ----------
        value : float or pandas.Series
            Value or series of values to be pivoted.

        Returns
        -------
        pivoted_value : float or pandas.Series
            Pivoted value or series.
        """

        logger.info('pivoting values')
        return 2 * self.pivot_value - value
