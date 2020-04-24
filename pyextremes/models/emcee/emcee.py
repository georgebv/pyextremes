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

    def __init__(
            self,
            extremes: pd.Series,
            distribution: typing.Union[str, scipy.stats.rv_continuous]
    ) -> None:
        self.emcee_distribution = None
        super().__init__(extremes=extremes, distribution=distribution)

    def fit(
            self,
            extremes: pd.Series,
            **kwargs: dict
    ) -> typing.Dict[str, np.ndarray]:
        n_walkers = kwargs.pop('n_walkers', 100)
        n_samples = kwargs.pop('n_samples', 500)
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        logger.info(f'getting distribution \'{self.distribution.name}\'')
        self.emcee_distribution = get_distribution(self.distribution.name, extremes=extremes)

        logger.info('defining emcee ensemble sampler')
        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.emcee_distribution.number_of_parameters,
            log_prob_fn=self.emcee_distribution.log_probability
        )

        logger.info(f'running the sampler with {n_walkers} walkers and {n_samples} samples')
        sampler.run_mcmc(
            initial_state=self.emcee_distribution.get_initial_state(n_walkers=n_walkers),
            nsteps=n_samples
        )

        logger.info(
            'calculating maximum aposteriori values of distribution paramters '
            'using kernel density estimation of peak'
        )
        map_estimate = np.zeros(self.emcee_distribution.number_of_parameters)
        for i in range(self.emcee_distribution.number_of_parameters):
            kde = scipy.stats.gaussian_kde(sampler.get_chain()[:, :, i].flatten())
            support = np.linspace(*np.quantile(sampler.get_chain()[:, :, i].flatten(), [0.025, 0.975]), 1000)
            density = kde.evaluate(support)
            map_estimate[i] = support[density.argmax()]

        return {
            'map': map_estimate,
            'trace': sampler.get_chain().transpose((1, 0, 2))
        }

    def retrieve_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs: dict
    ) -> tuple:
        burn_in = kwargs.pop('burn_in')
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        logger.debug('checking if the result has been previously hashed in self.hashed_return_values')
        condition_1 = exceedance_probability not in self.hashed_return_values
        if condition_1:
            condition_2 = True
        else:
            condition_2 = alpha not in self.hashed_return_values[exceedance_probability]
        if condition_2:
            condition_3 = True
        else:
            condition_3 = burn_in not in self.hashed_return_values[exceedance_probability][alpha]
        if condition_1 or condition_2 or condition_3:
            logger.debug('full entry not hashed, calculating return value')
            rv = self.__get_return_value(exceedance_probability=exceedance_probability, alpha=alpha, burn_in=burn_in)
            if condition_1:
                logger.debug(f'creating a new entry for exceedance_probability {exceedance_probability}')
                self.hashed_return_values[exceedance_probability] = {
                    'return value': rv[0],
                    alpha: {
                        burn_in: rv[1]
                    }
                }
            elif condition_2:
                logger.debug(f'updating entry for exceedance_probability {exceedance_probability} with alpha {alpha}')
                self.hashed_return_values[exceedance_probability][alpha] = {
                    burn_in: rv[1]
                }
            elif condition_3:
                logger.debug(
                    f'updating entry for exceedance_probability {exceedance_probability} and alpha {alpha} '
                    f'with burn_in {burn_in}'
                )
                self.hashed_return_values[exceedance_probability][alpha][burn_in] = rv[1]

        logger.debug('retrieving hashed entry and returning results')
        hashed_entry = self.hashed_return_values[exceedance_probability]
        return (hashed_entry['return value'], *hashed_entry[alpha][burn_in])

    def __get_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            burn_in: int
    ) -> tuple:
        if not isinstance(burn_in, int):
            raise TypeError(f'invalid type in {type(burn_in)} for the \'burn_in\' argument')
        if burn_in < 0:
            raise ValueError(f'\'{burn_in}\' is not a valid \'burn_in\' value')
        if burn_in >= burn_in >= self.fit_parameters['trace'].shape[1]:
            raise ValueError(
                f'\'burn_in\' value \'{burn_in}\' exceeds number of samples {self.fit_parameters["trace"].shape[1]}'
            )

        return_value = self.emcee_distribution.isf(q=exceedance_probability, parameters=self.fit_parameters['map'])
        if alpha is None:
            confidence_interval = (None, None)
        else:
            rv_sample = []
            for walker_trace in self.fit_parameters['trace']:
                for parameters in walker_trace[burn_in:]:
                    rv_sample.append(
                        self.emcee_distribution.isf(q=exceedance_probability, parameters=parameters)
                    )
            confidence_interval = tuple(np.quantile(a=rv_sample, q=np.array([(1-alpha)/2, (1+alpha)/2])))
        return return_value, confidence_interval


if __name__ == '__main__':
    # Crete extreme value files to be used for tests
    pass
    import os
    import pathlib
    test_data_folder = pathlib.Path(os.path.realpath(__file__)).parent / 'tests' / 'data'
    test_extremes = pd.read_csv(test_data_folder / 'extremes_bm_high.csv', index_col=0, parse_dates=True, squeeze=True)

    self = Emcee(extremes=test_extremes, distribution='genextreme')
