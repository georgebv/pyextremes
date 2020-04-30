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


class GumbelR(AbstractEmceeDistributionBaseClass):

    def _fit(self) -> tuple:
        return scipy.stats.gumbel_r.fit(self.extremes)

    @property
    def number_of_parameters(self) -> int:
        return 2

    def log_prior(
            self,
            theta: tuple
    ) -> float:
        location, scale = theta
        location_prior = scipy.stats.norm.logpdf(x=location, loc=self.mle_parameters[0], scale=100)
        scale_prior = scipy.stats.norm.logpdf(x=scale, loc=self.mle_parameters[1], scale=100) if scale > 0 else -np.inf
        return location_prior + scale_prior

    def log_likelihood(
            self,
            theta: tuple
    ) -> float:
        location, scale = theta

        # Parameter constraint
        if scale <= 0:
            return -np.inf

        # Calculate log-likelihood
        return sum(scipy.stats.gumbel_r.logpdf(x=self.extremes.values, loc=location, scale=scale))

    def _get_prop(
            self,
            prop: str,
            x: typing.Union[float, np.ndarray],
            parameters: tuple
    ) -> typing.Union[float, np.ndarray]:
        return getattr(scipy.stats.gumbel_r, prop)(x, loc=parameters[0], scale=parameters[1])
