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

import emcee
import numpy as np
import pandas as pd

from pyextremes.models.emcee.distributions import get_distribution
from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)


class Emcee(AbstractModelBaseClass):

    def fit(
            self,
            extremes: pd.Series,
            **kwargs: dict
    ) -> dict:
        n_walkers = kwargs.pop('n_walkers', 100)
        n_samples = kwargs.pop('n_samples', 500)
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        distribution = get_distribution(self.distribution.name, extremes=extremes)
        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=distribution.number_of_parameters,
            log_prob_fn=distribution.log_probability
        )
        sampler.run_mcmc(
            initial_state=distribution.get_initial_state(n_walkers=n_walkers),
            nsteps=n_samples
        )
        return {
            'map': np.nan,  # TODO - find as combination with highest likelihood
            'trace': sampler.get_chain()
        }
