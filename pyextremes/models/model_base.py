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
                self.distribution = getattr(scipy.stats, distribution)
            except AttributeError:
                raise ValueError(f'\'{distribution}\' is not a valid \'distribution\' value')
        elif isinstance(distribution, scipy.stats.rv_continuous):
            self.distribution = distribution
        else:
            raise TypeError(f'invalid type in {type(distribution)} for the \'distribution\' argument')

        logger.info('calling the self.fit method')
        self.fit_parameters = self.fit(extremes=extremes)

        logger.info('initializing the return value hash')
        self.hashed_return_values = {}

    @abc.abstractmethod
    def fit(
            self,
            extremes: pd.Series,
            **kwargs: dict
    ) -> typing.Union[typing.Tuple[float], typing.Dict[str, np.ndarray]]:
        pass

    def get_return_value(
            self,
            exceedance_probability: float,
            alpha: float = None,
            **kwargs: dict
    ) -> typing.Union[tuple, np.ndarray]:
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
            Optional keyword arguments passed to a model.

        Returns
        -------
        return_value : float or array-like
            Return value(s).
        lower_ci_bound : float or array-like
            Lower confidence interval bound(s).
        upper_ci_bount : float or array-like
            Upper confidence interval bound(s).
        """

        if isinstance(exceedance_probability, (np.ndarray, list)):
            logger.info('getting a list of return values')
            return np.transpose(
                [
                    self.retrieve_return_value(exceedance_probability=ep, alpha=alpha, **kwargs)
                    for ep in exceedance_probability
                ]
            ).astype(float)
        elif isinstance(exceedance_probability, float):
            logger.info('getting a single return value')
            return self.retrieve_return_value(exceedance_probability=exceedance_probability, alpha=alpha, **kwargs)
        else:
            raise TypeError(
                f'invalid type in {type(exceedance_probability)} for the \'exceedance_probability\' argument'
            )

    @abc.abstractmethod
    def retrieve_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs: dict
    ) -> tuple:
        pass
