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

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)


class AbstractModelBaseClass(abc.ABC):

    def __init__(
            self,
            extremes: pd.Series,
            distribution: typing.Union[str, scipy.stats.rv_continuous]
    ) -> None:
        logger.info('getting extreme value distribution')
        if isinstance(distribution, str):
            try:
                logger.info(f'getting distribution object with name \'{distribution}\' from scipy.stats')
                self.extremes = getattr(scipy.stats, distribution)
            except AttributeError:
                raise ValueError(f'\'{distribution}\' is not a valid \'distribution\' value')
        elif isinstance(distribution, scipy.stats.rv_continuous):
            self.distribution = distribution
        else:
            raise TypeError(f'invalid type in {type(distribution)} for the \'distribution\' argument')

        logger.info('calling the self.__fit function from the model base model class __init__ method')
        self.fit_parameters = self.fit(extremes=extremes)

        logger.info('initializing the return value hash')
        self.__return_value_hash = {}

    @abc.abstractmethod
    def fit(
            self,
            extremes: pd.Series,
            **kwargs: dict
    ) -> typing.Union[tuple, dict]:
        pass

    def get_return_value(
            self,
            return_period: typing.Union[np.ndarray, str, typing.List[str], pd.Timedelta, typing.List[pd.Timedelta]],
            alpha: float = None,
            n_samples: int = None
    ) -> typing.Union[tuple, np.ndarray]:
        logger.info('calling self.get_return_value of the base model class')
        if isinstance(return_period, (np.ndarray, list)):
            logger.info('getting a list of return values')
            return np.array(
                [self.__retrieve_return_value(rp, alpha=alpha, n_samples=n_samples) for rp in return_period]
            )
        elif isinstance(return_period, (str, pd.Timedelta)):
            logger.info('getting a single return value')
            return self.__retrieve_return_value(return_period, alpha=alpha, n_samples=n_samples)
        else:
            raise TypeError(f'invalid type in {type(return_period)} for the \'return_period\' argument')

    def __retrieve_return_value(
            self,
            return_period: typing.Union[str, pd.Timedelta],
            alpha: float,
            n_samples: int
    ) -> tuple:
        logger.debug('checking if the result has been previously hashed in self.__return_value_hash')
        flag = False
        if return_period not in self.__return_value_hash:
            logger.debug(f'return_period {return_period} not hashed')
            flag = True
        elif alpha not in self.__return_value_hash['return value']:
            logger.debug(f'alpha {alpha} not hashed')
            flag = True
        elif n_samples not in self.__return_value_hash['return value'][alpha]:
            logger.debug(f'n_samples {n_samples} not hashed')
            flag = True

        if flag:
            logger.debug('calculating return value for new parameters and adding it to hashed results')
            rv = self.__get_return_value(return_period=return_period, alpha=alpha, n_samples=n_samples)
            self.__return_value_hash[return_period] = {
                'return value': rv[0],
                alpha: {
                    n_samples: rv[1]
                }
            }

        logger.debug('retrieving hashed entry and returning results')
        hashed_entry = self.__return_value_hash[return_period]
        return hashed_entry['return value'], hashed_entry[alpha]

    @abc.abstractmethod
    def __get_return_value(
            self,
            return_period: typing.Union[str, pd.Timedelta],
            alpha: float,
            n_samples: int
    ) -> tuple:
        pass
