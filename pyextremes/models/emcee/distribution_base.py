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


class AbstractEmceeDistributionBaseClass(abc.ABC):

    def __init__(
            self,
            extremes: pd.Series
    ) -> None:
        self.extremes = extremes
        logger.info('calling the fit method')
        self.mle_parameters = self.fit()

    @abc.abstractmethod
    def fit(self) -> tuple:
        """
        Find model parameters using scipy MLE fit method.

        Returns
        -------
        Model parameters.
        """
        pass

    @property
    @abc.abstractmethod
    def number_of_parameters(self) -> int:
        pass

    @abc.abstractmethod
    def log_prior(
            self,
            theta: tuple
    ) -> float:
        pass

    @abc.abstractmethod
    def log_likelihood(
            self,
            theta: tuple
    ) -> float:
        pass

    def log_probability(
            self,
            theta: tuple
    ) -> float:
        return self.log_prior(theta=theta) + self.log_likelihood(theta=theta)

    def get_initial_state(
            self,
            n_walkers: int
    ) -> np.ndarray:
        """
        Get initial positions of emcee sampler walkers.
        Positions are sampled from a normal distribution for each of the model parameters
        (e.g. shape, location, and scale) with location being derived from scipy MLE fit
        and standard deviation being 0.01.

        Parameters
        ----------
        n_walkers : int
            Number of walkers used by the emcee sampler.

        Returns
        -------
        initial_positions : numpy.ndarray
            Array with initial positions of emcee sampler walkers.
        """

        logger.info(f'getting initial positions for {n_walkers} walkers')
        return np.transpose(
            [
                scipy.stats.norm.rvs(loc=parameter, scale=0.01, size=n_walkers)
                for parameter in self.mle_parameters
            ]
        )

    @abc.abstractmethod
    def isf(
            self,
            q: typing.Union[float, np.ndarray],
            parameters: tuple
    ) -> typing.Union[float, np.ndarray]:
        pass
