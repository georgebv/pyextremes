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

from pyextremes.models.emcee.distribution_base import AbstractEmceeDistributionBaseClass

logger = logging.getLogger(__name__)


class AbstractModelBaseClass(abc.ABC):

    def __init__(
            self,
            extremes: pd.Series,
            distribution: str,
            **kwargs
    ) -> None:
        """
        Distribution model base class.

        Parameters
        ----------
        extremes : pandas.Series
            Time series of transformed extreme events.
        distribution : str
            Name of scipy.stats distribution.
        kwargs : dict
            Keyword arguments passed to a model .__fit method.
            MLE model:
                TODO
            Emcee model:
                n_walkers : int
                    The number of walkers in the ensemble.
                n_samples : int
                    The number of steps to run.
        """

        self.extremes = extremes

        logger.info('getting extreme value distribution')
        self.distribution = self.__get_distribution(distribution=distribution)

        logger.info('fitting the distribution to extremes')
        self.fit_parameters = self.__fit(extremes=extremes, **kwargs)

        logger.info('initializing the return value hash')
        self.hashed_return_values = {}

    @abc.abstractmethod
    def __get_distribution(
            self,
            distribution: str
    ) -> typing.Union[scipy.stats.rv_continuous, AbstractEmceeDistributionBaseClass]:
        pass

    @abc.abstractmethod
    def __fit(
            self,
            extremes: pd.Series,
            **kwargs: dict
    ) -> typing.Union[tuple, dict]:
        pass

    @staticmethod
    @abc.abstractmethod
    def __decode_kwargs(kwargs: dict) -> str:
        pass

    def get_return_value(
            self,
            exceedance_probability: typing.Union[float, typing.Iterable[float]],
            alpha: float = None,
            **kwargs: dict
    ) -> tuple:
        """
        Get return value and confidence interval for a given exceedance probability.

        Parameters
        ----------
        exceedance_probability : float or array-like
            Exceedance probability or array of exceedance probabilities.
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=None).
            If None, return None for upper and lower confidence interval bounds.
        kwargs : dict
            Keyword arguments passed to a model .__get_return_value method.
            MLE model:
                TODO
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        return_value : float or array-like
            Return value(s).
        lower_ci_bound : float or array-like
            Lower confidence interval bound(s).
        upper_ci_bount : float or array-like
            Upper confidence interval bound(s).
        """

        if hasattr(exceedance_probability, '__iter__') and not isinstance(exceedance_probability, str):
            logger.info('getting a list of return values')
            return tuple(
                np.transpose(
                    [
                        self.__retrieve_return_value(exceedance_probability=ep, alpha=alpha, **kwargs)
                        for ep in exceedance_probability
                    ]
                ).astype(float)
            )
        elif isinstance(exceedance_probability, float):
            logger.info('getting a single return value')
            return self.__retrieve_return_value(exceedance_probability=exceedance_probability, alpha=alpha, **kwargs)
        else:
            raise TypeError(
                f'invalid type in {type(exceedance_probability)} for the \'exceedance_probability\' argument'
            )

    def __retrieve_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs: dict
    ) -> tuple:
        """
        Retrieve return value and confidence interval from hashed results dictionary.
        If the return value has not been previously hashed, calculate it, add it to hash, and return.

        Parameters
        ----------
        exceedance_probability : float
            Exceedance probability.
        alpha : float
            Width of confidence interval, from 0 to 1 (default=None).
            If None, return None for upper and lower confidence interval bounds.
        kwargs : dict
            Keyword arguments passed to a model .__get_return_value method.
            MLE model:
                TODO
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        return_value : float
        lower_confidence_interval_bound : float
        upper_confidence_interval_bound : float
        """

        logger.debug('decoding kwargs')
        decoded_kwargs = self.__decode_kwargs(kwargs=kwargs)

        try:
            logger.debug(
                f'trying to retrieve result from hash for exceedance_probability {exceedance_probability:.6f}, '
                f'alpha {alpha:.6f}, and kwargs {decoded_kwargs}'
            )
            hashed_entry = self.hashed_return_values[f'{exceedance_probability:.6f}']
            return_value = hashed_entry['return value']
            confidence_interval = hashed_entry[f'{alpha:.6f}'][decoded_kwargs]
            logger.debug('successfully retrieved result from has - returning values')
            return (return_value, *confidence_interval)
        except KeyError:
            logger.debug(
                f'calculating new results for exceedance_probability {exceedance_probability:.6f}, '
                f'alpha {alpha:.6f}, and kwargs {decoded_kwargs}'
            )
            rv = self.__get_return_value(exceedance_probability=exceedance_probability, alpha=alpha, **kwargs)
            if f'{exceedance_probability:.6f}' in self.hashed_return_values:
                if f'{alpha:.6f}' in self.hashed_return_values[f'{exceedance_probability:.6f}']:
                    logger.debug(
                        f'updating entry for exceedance_probability {exceedance_probability:.6f} '
                        f'and alpha {alpha:.6f} with kwargs {decoded_kwargs}'
                    )
                    self.hashed_return_values[f'{exceedance_probability:.6f}'][f'{alpha:.6f}'][decoded_kwargs] = rv[1]
                else:
                    logger.debug(
                        f'updating entry for exceedance_probability {exceedance_probability:.6f} '
                        f'with alpha {alpha:.6f} and kwargs {decoded_kwargs}'
                    )
                    self.hashed_return_values[f'{exceedance_probability:.6f}'][f'{alpha:.6f}'] = {
                        decoded_kwargs: rv[1]
                    }
            else:
                logger.debug(
                    f'creating a new entry for exceedance_probability {exceedance_probability:.6f}, '
                    f'alpha {alpha:.6f}, and kwargs {decoded_kwargs}'
                )
                self.hashed_return_values[f'{exceedance_probability:.6f}'] = {
                    'return value': rv[0],
                    f'{alpha:.6f}': {
                        decoded_kwargs: rv[1]
                    }
                }
            return self.__retrieve_return_value(
                exceedance_probability=exceedance_probability,
                alpha=alpha,
                **kwargs
            )

    @abc.abstractmethod
    def __get_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs: dict
    ) -> tuple:
        pass
