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
import pandas as pd
import scipy.stats

from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)


class MLE(AbstractModelBaseClass):
    """
    Maximum Likelihood Estimate (MLE) model built around the scipy.stats package.
    """

    def _get_distribution(
            self,
            distribution: str
    ) -> scipy.stats.rv_continuous:
        return getattr(scipy.stats, distribution)

    def _fit(
            self,
            extremes: pd.Series,
            **kwargs
    ) -> tuple:
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
        distributions = ['genpareto', 'expon', 'genextreme', 'gumbel_r']
        if self.distribution.name in ['genpareto', 'expon']:
            return self.distribution.fit(extremes, floc=0)
        elif self.distribution.name in ['genextreme', 'gumbel_r']:
            return self.distribution.fit(extremes)
        else:
            raise NotImplementedError(
                f'\'{self.distribution.name}\' distribution is not implemented for the \'MLE\' model\n'
                f'Available \'MLE\' distributions: {", ".join(distributions)}'
            )

    def _decode_kwargs(
            self,
            kwargs: dict
    ) -> str:
        n_samples = kwargs['n_samples']
        if not isinstance(n_samples, int):
            raise TypeError(f'invalid type in {type(n_samples)} for the \'n_samples\' argument')
        if n_samples <= 0:
            raise ValueError(f'\'{n_samples}\' is not a valid \'n_samples\' value, it must be a positive integer')
        return f'{n_samples:d}'

    def _get_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs
    ) -> tuple:
        logger.debug('calculating return value')
        return_value = self.distribution.isf(exceedance_probability, *self.fit_parameters)
        if alpha is None:
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            logger.debug('returning confidence interval as None for alpha=None')
            confidence_interval = (None, None)
        else:
            n_samples = kwargs.pop('n_samples')
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            def draw_sample():
                sample_size = scipy.stats.poisson.rvs(mu=len(self.extremes), loc=0, size=None)
                sample = np.random.choice(a=self.extremes.values, size=sample_size, replace=True)
                sample_fit_parameters = self._fit(extremes=sample)
                return self.distribution.isf(exceedance_probability, *sample_fit_parameters)

            logger.info('drawing return value samples from the extremes')
            rv_sample = np.array([draw_sample() for _ in range(n_samples)])

            logger.debug('calculating confidence interval')
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
        return self.distribution.pdf(x, *self.fit_parameters)

    def cdf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.distribution.cdf(x, *self.fit_parameters)
