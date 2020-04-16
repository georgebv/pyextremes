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

import abc
import logging
import typing

import pandas as pd
import scipy.stats

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class AbstractDistributionFittingModel(abc.ABC):

    def __init__(
            self,
            extremes: pd.Series,
            distribution: typing.Union[str, scipy.stats.rv_continuous]
    ):
        self.extremes = extremes

        logger.info('getting extreme value distribution')
        if isinstance(distribution, str):
            try:
                self.extremes = getattr(scipy.stats, distribution)
            except AttributeError:
                raise ValueError(f'{distribution} is not a valid distribution value')
        elif isinstance(distribution, scipy.stats.rv_continuous):
            self.distribution = distribution
        else:
            raise TypeError(f'invalid type in {type(distribution)} for the distribution argument')

        logger.info('calling the fit self.function from the model __init__ method')
        self.fit()

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def get_return_value(
            self,
            return_period: typing.Union[str, pd.Timedelta] = '1Y'
    ):
        pass

    @abc.abstractmethod
    def get_return_period(
            self,
            return_value: typing.Union[int, float]
    ):
        pass

    @abc.abstractmethod
    def get_confidence_interval(
            self,
            return_period: typing.Union[str, pd.Timedelta] = '1Y',
            alpha: float = 0.95
    ):
        pass
