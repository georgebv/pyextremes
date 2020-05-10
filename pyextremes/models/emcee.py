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
import warnings

import emcee
import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)


class Emcee(AbstractModelBaseClass):
    """
    Markov Chain Monte Carlo (MCMC) model.
    Built around the emcee package by Daniel Foreman-Mackey
    """

    def __init__(
            self,
            extremes: pd.Series,
            distribution: typing.Union[str, scipy.stats.rv_continuous],
            distribution_kwargs: dict = None,
            n_walkers: int = 100,
            n_samples: int = 500,
            progress: bool = False
    ) -> None:
        super().__init__(
            extremes=extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            n_walkers=n_walkers,
            n_samples=n_samples,
            progress=progress
        )

    @property
    def name(self) -> str:
        return 'Emcee'

    def fit(
            self,
            n_walkers: int,
            n_samples: int,
            progress: bool
    ) -> None:
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
            'by finding peaks of corresponding PDFs using Gaussian kernel density estimation. '
            'One third of samples is discarded as burn-in.'
        )
        map_estimate = np.zeros(self.distribution.number_of_parameters)
        for i in range(self.distribution.number_of_parameters):
            parameter_values = sampler.get_chain()[n_samples//3:, :, i].flatten()
            kde = scipy.stats.gaussian_kde(dataset=parameter_values)
            support = np.linspace(
                *np.quantile(parameter_values, [0.025, 0.975]),
                1000
            )
            density = kde.evaluate(support)
            map_estimate[i] = support[density.argmax()]

        logger.info('setting fit parameters and trace')
        self.fit_parameters = dict(zip(self.distribution.free_parameters, map_estimate))
        self.trace = sampler.get_chain().transpose((1, 0, 2))

    def _encode_kwargs(self, kwargs: dict) -> str:
        burn_in = kwargs['burn_in']
        if not isinstance(burn_in, int):
            raise TypeError(
                f'invalid type in {type(burn_in)} for the \'burn_in\' argument, it must be a positive integer'
            )
        if burn_in < 0:
            raise ValueError(f'\'{burn_in}\' is not a valid \'burn_in\' value, it must be a positive integer')
        if burn_in >= self.trace.shape[1]:
            raise ValueError(f'\'burn_in\' value \'{burn_in}\' exceeds number of samples {self.trace.shape[1]}')
        return f'{burn_in:d}'

    def _get_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs
    ) -> tuple:
        logger.debug('calculating return value')
        return_value = self.distribution.distribution.isf(
            q=exceedance_probability,
            **self.fit_parameters,
            **self.distribution._fixed_parameters
        )

        if alpha is None:
            if 'burn_in' in kwargs:
                kwargs.pop('burn_in')
                warnings.warn(message='burn_in is not used when alpha is None')
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            logger.debug('returning confidence interval as None for alpha=None')
            confidence_interval = (None, None)

        else:
            burn_in = kwargs.pop('burn_in')
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            logger.debug('calculating confidence interval')
            rv_sample = self.distribution.get_prop(
                prop='isf',
                x=exceedance_probability,
                free_parameters=np.vstack(self.trace[:, burn_in:, :])
            )
            confidence_interval = tuple(
                np.quantile(
                    a=rv_sample,
                    q=[(1-alpha)/2, (1+alpha)/2]
                )
            )

        logger.debug('returning return value and confidence interval')
        return return_value, confidence_interval
