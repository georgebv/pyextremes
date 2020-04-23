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

import numpy as np
import scipy.stats

from pyextremes.models.emcee.distribution_base import AbstractEmceeDistributionBaseClass

logger = logging.getLogger(__name__)


class Genpareto(AbstractEmceeDistributionBaseClass):

    def fit(self) -> tuple:
        shape, location, scale = scipy.stats.genpareto.fit(self.extremes, floc=0)
        return shape, scale

    @property
    def number_of_parameters(self) -> int:
        return 2

    def log_prior(
            self,
            theta: tuple
    ) -> float:
        shape, scale = theta
        shape_prior = scipy.stats.norm.logpdf(x=shape, loc=self.mle_parameters[0], scale=100)
        location_prior = 0
        scale_prior = scipy.stats.norm.logpdf(x=scale, loc=self.mle_parameters[2], scale=100) if scale > 0 else -np.inf
        return shape_prior + location_prior + scale_prior

    def log_likelihood(
            self,
            theta: tuple
    ) -> float:
        shape, scale = theta

        # Parameter constraint
        if scale <= 0:
            return -np.inf

        # Support constraint
        if shape >= 0:
            condition = np.all(self.extremes >= 0)
        else:
            condition = np.all(self.extremes >= 0) and np.all(self.extremes >= scale / shape)

        # Calculate log-likelihood
        if condition:
            return sum(scipy.stats.genpareto.logpdf(x=self.extremes, c=shape, loc=0, scale=scale))
        else:
            return -np.inf
