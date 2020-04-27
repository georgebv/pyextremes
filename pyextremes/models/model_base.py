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

from pyextremes.models.emcee.distributions.distribution_base import AbstractEmceeDistributionBaseClass

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
        kwargs
            Keyword arguments passed to a model ._fit method.
            MLE model:
                MLE model takes no additional arguments.
            Emcee model:
                n_walkers : int
                    The number of walkers in the ensemble.
                n_samples : int
                    The number of steps to run.
        """

        self.extremes = extremes

        logger.info('getting extreme value distribution')
        self.distribution = self._get_distribution(distribution=distribution)

        logger.info('fitting the distribution to extremes')
        self.fit_parameters = self._fit(extremes=extremes, **kwargs)

        logger.info('initializing the return value hash')
        self.hashed_return_values = {}

    @abc.abstractmethod
    def _get_distribution(
            self,
            distribution: str
    ) -> typing.Union[scipy.stats.rv_continuous, AbstractEmceeDistributionBaseClass]:
        pass

    @abc.abstractmethod
    def _fit(
            self,
            extremes: pd.Series,
            **kwargs
    ) -> typing.Union[tuple, dict]:
        pass

    @abc.abstractmethod
    def _decode_kwargs(
            self,
            kwargs: dict
    ) -> str:
        pass

    @abc.abstractmethod
    def _test_kwargs(
            self,
            kwargs: dict
    ) -> None:
        pass

    def get_return_value(
            self,
            exceedance_probability: typing.Union[float, typing.Iterable[float]],
            alpha: float = None,
            **kwargs
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
        kwargs
            Keyword arguments passed to a model ._get_return_value method.
            If alpha is None, no keyword arguments are required or accepted.
            MLE model:
                n_samples : int
                    Number of samles used to get confidence interval.
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
                        self._retrieve_return_value(exceedance_probability=ep, alpha=alpha, **kwargs)
                        for ep in exceedance_probability
                    ]
                ).astype(float)
            )
        elif isinstance(exceedance_probability, float):
            logger.info('getting a single return value')
            return self._retrieve_return_value(exceedance_probability=exceedance_probability, alpha=alpha, **kwargs)
        else:
            raise TypeError(
                f'invalid type in {type(exceedance_probability)} for the \'exceedance_probability\' argument'
            )

    def _retrieve_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs
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
        kwargs
            Keyword arguments passed to a model ._get_return_value method.
            If alpha is None, no keyword arguments are required or accepted.

        Returns
        -------
        return_value : float
        lower_confidence_interval_bound : float
        upper_confidence_interval_bound : float
        """

        if alpha is None:
            logger.debug('alpha is None - setting kwargs and alpha to None')
            decoded_kwargs = 'None'
            decoded_alpha = 'None'
        else:
            logger.debug('testing kwargs validity')
            self._test_kwargs(kwargs=kwargs)

            logger.debug('decoding alpha and kwargs')
            decoded_kwargs = self._decode_kwargs(kwargs=kwargs)
            decoded_alpha = f'{alpha:.6f}'

        try:
            logger.debug(
                f'trying to retrieve result from hash for exceedance_probability {exceedance_probability:.6f}, '
                f'alpha {decoded_alpha}, and kwargs {decoded_kwargs}'
            )
            hashed_entry = self.hashed_return_values[f'{exceedance_probability:.6f}']
            return_value = hashed_entry['return value']
            confidence_interval = hashed_entry[decoded_alpha][decoded_kwargs]
            logger.debug('successfully retrieved result from has - returning values')
            return (return_value, *confidence_interval)
        except KeyError:
            logger.debug(
                f'calculating new results for exceedance_probability {exceedance_probability:.6f}, '
                f'alpha {decoded_alpha}, and kwargs {decoded_kwargs}'
            )
            rv = self._get_return_value(exceedance_probability=exceedance_probability, alpha=alpha, **kwargs)
            if f'{exceedance_probability:.6f}' in self.hashed_return_values:
                if decoded_alpha in self.hashed_return_values[f'{exceedance_probability:.6f}']:
                    logger.debug(
                        f'updating entry for exceedance_probability {exceedance_probability:.6f} '
                        f'and alpha {decoded_alpha} with kwargs {decoded_kwargs}'
                    )
                    self.hashed_return_values[f'{exceedance_probability:.6f}'][decoded_alpha][decoded_kwargs] = rv[1]
                else:
                    logger.debug(
                        f'updating entry for exceedance_probability {exceedance_probability:.6f} '
                        f'with alpha {decoded_alpha} and kwargs {decoded_kwargs}'
                    )
                    self.hashed_return_values[f'{exceedance_probability:.6f}'][decoded_alpha] = {
                        decoded_kwargs: rv[1]
                    }
            else:
                logger.debug(
                    f'creating a new entry for exceedance_probability {exceedance_probability:.6f}, '
                    f'alpha {decoded_alpha}, and kwargs {decoded_kwargs}'
                )
                self.hashed_return_values[f'{exceedance_probability:.6f}'] = {
                    'return value': rv[0],
                    decoded_alpha: {
                        decoded_kwargs: rv[1]
                    }
                }
            return self._retrieve_return_value(
                exceedance_probability=exceedance_probability,
                alpha=alpha,
                **kwargs
            )

    @abc.abstractmethod
    def _get_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs
    ) -> tuple:
        pass
