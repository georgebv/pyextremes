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

import emcee
import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.models.emcee.distributions import get_distribution
from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)


class Emcee(AbstractModelBaseClass):

    def fit(
            self,
            extremes: pd.Series,
            **kwargs: dict
    ) -> typing.Dict[str, np.ndarray]:
        n_walkers = kwargs.pop('n_walkers', 100)
        n_samples = kwargs.pop('n_samples', 500)
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        logger.info(f'getting distribution \'{self.distribution.name}\'')
        distribution = get_distribution(self.distribution.name, extremes=extremes)

        logger.info('defining emcee ensemble sampler')
        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=distribution.number_of_parameters,
            log_prob_fn=distribution.log_probability
        )

        logger.info(f'running the sampler with {n_walkers} walkers and {n_samples} samples')
        sampler.run_mcmc(
            initial_state=distribution.get_initial_state(n_walkers=n_walkers),
            nsteps=n_samples
        )

        logger.info(
            'calculating maximum aposteriori values of distribution paramters '
            'using kernel density estimation of peak'
        )
        map_estimate = np.zeros(distribution.mle_parameters)
        for i in range(distribution.number_of_parameters):
            kde = scipy.stats.gaussian_kde(sampler.get_chain()[:, :, i].flatten())
            support = np.linspace(*np.quantile(sampler.get_chain()[:, :, i].flatten(), [0.025, 0.975]), 1000)
            density = kde.evaluate(support)
            map_estimate[i] = support[density.argmax()]

        return {
            'map': map_estimate,
            'trace': sampler.get_chain().transpose((1, 0, 2))
        }


if __name__ == '__main__':
    # Crete extreme value files to be used for tests
    pass
    # import os
    # import pathlib
    # test_data_folder = pathlib.Path(os.path.realpath(__file__)).parent / 'tests' / 'data'
    # extremes = pd.read_csv(test_data_folder / 'extremes_bm_high.csv', index_col=0, parse_dates=True, squeeze=True)
    # kwargs = {}
    # distribution = get_distribution('genextreme', extremes=extremes)
