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

import numpy as np
import scipy.stats

from pyextremes.models.emcee.distributions.distribution_base import AbstractEmceeDistributionBaseClass

logger = logging.getLogger(__name__)


class Expon(AbstractEmceeDistributionBaseClass):

    def _fit(self) -> tuple:
        location, scale = scipy.stats.expon.fit(self.extremes, floc=0)
        return (scale,)

    @property
    def number_of_parameters(self) -> int:
        return 1

    def log_prior(
            self,
            theta: tuple
    ) -> float:
        scale = theta[0]
        location_prior = 0
        scale_prior = scipy.stats.norm.logpdf(x=scale, loc=self.mle_parameters[0], scale=100) if scale > 0 else -np.inf
        return location_prior + scale_prior

    def log_likelihood(
            self,
            theta: tuple
    ) -> float:
        scale = theta[0]

        # Parameter constraint
        if scale <= 0:
            return -np.inf

        # Support constraint
        condition = np.all(self.extremes.values >= 0)

        # Calculate log-likelihood
        if condition:
            return sum(scipy.stats.expon.logpdf(x=self.extremes.values, loc=0, scale=scale))
        else:
            return -np.inf

    def _get_prop(
            self,
            prop: str,
            x: typing.Union[float, np.ndarray],
            parameters: tuple
    ) -> typing.Union[float, np.ndarray]:
        return getattr(scipy.stats.expon, prop)(x, loc=0, scale=parameters[0])
