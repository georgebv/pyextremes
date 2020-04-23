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
        self.mle_parameters = self.fit()

    @abc.abstractmethod
    def fit(self) -> tuple:
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
        return np.transpose(
            [
                scipy.stats.norm.rvs(loc=parameter, scale=1, size=n_walkers)
                for parameter in self.mle_parameters
            ]
        )
