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

from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)


class Emcee(AbstractModelBaseClass):
    """
    Markov Chain Monte Carlo (MCMC) model built around the emcee package by Daniel Foreman-Mackey.
    """

    @property
    def name(self) -> str:
        return 'Emcee'

    def _fit(
            self,
            extremes: pd.Series,
            **kwargs
    ) -> dict:
        n_walkers = kwargs.pop('n_walkers', 100)
        n_samples = kwargs.pop('n_samples', 500)
        progress = kwargs.pop('progress', False)
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        logger.info('defining emcee ensemble sampler')
        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.distribution.number_of_parameters,
            log_prob_fn=self.distribution.log_probability
        )

        logger.info(f'running the sampler with {n_walkers} walkers and {n_samples} samples')
        sampler.run_mcmc(
            initial_state=self.distribution.get_initial_state(n_walkers=n_walkers),
            nsteps=n_samples, progress=progress
        )

        logger.info(
            'calculating maximum aposteriori values of distribution paramters '
            'by finding peaks of corresponding PDF\'s using gaussian kernel density estimation. '
            'One third of samples as discarded as burn-in.'
        )
        map_estimate = np.zeros(self.distribution.number_of_parameters)
        for i in range(self.distribution.number_of_parameters):
            kde = scipy.stats.gaussian_kde(sampler.get_chain()[n_samples//3:, :, i].flatten())
            support = np.linspace(
                *np.quantile(sampler.get_chain()[n_samples//3:, :, i].flatten(), [0.025, 0.975]),
                1000
            )
            density = kde.evaluate(support)
            map_estimate[i] = support[density.argmax()]

        return {
            'map': map_estimate,
            'trace': sampler.get_chain().transpose((1, 0, 2))
        }

    @property
    def loglikelihood(self) -> float:
        return self.distribution.log_likelihood(theta=self.fit_parameters['map'])

    @property
    def AIC(self) -> float:
        return 2 * self.distribution.number_of_parameters - 2 * self.loglikelihood

    def _decode_kwargs(
            self,
            kwargs: dict
    ) -> str:
        burn_in = kwargs['burn_in']
        if not isinstance(burn_in, int):
            raise TypeError(f'invalid type in {type(burn_in)} for the \'burn_in\' argument')
        if burn_in < 0:
            raise ValueError(f'\'{burn_in}\' is not a valid \'burn_in\' value, it must be a positive integer')
        if burn_in >= self.fit_parameters['trace'].shape[1]:
            raise ValueError(
                f'\'burn_in\' value \'{burn_in}\' exceeds number of samples {self.fit_parameters["trace"].shape[1]}'
            )
        return f'{burn_in:d}'

    def _get_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs
    ) -> tuple:
        logger.debug('calculating return value')
        return_value = self.distribution.isf(q=exceedance_probability, parameters=self.fit_parameters['map'])
        if alpha is None:
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            logger.debug('returning confidence interval as None for alpha=None')
            confidence_interval = (None, None)
        else:
            burn_in = kwargs.pop('burn_in')
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            logger.debug('calculating confidence interval')
            rv_sample = self.distribution.isf(
                q=exceedance_probability,
                parameters=np.transpose(np.vstack(self.fit_parameters['trace'][:, burn_in:, :]))
            )
            confidence_interval = tuple(
                np.quantile(
                    a=rv_sample,
                    q=[(1-alpha)/2, (1+alpha)/2]
                )
            )
        return return_value, confidence_interval

    def pdf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.distribution.pdf(x=x, parameters=self.fit_parameters['map'])

    def cdf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.distribution.cdf(x=x, parameters=self.fit_parameters['map'])

    def ppf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.distribution.ppf(q=x, parameters=self.fit_parameters['map'])

    def isf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.distribution.isf(q=x, parameters=self.fit_parameters['map'])
