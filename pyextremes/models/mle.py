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
import warnings

import numpy as np
import pandas as pd

from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)
DEFAULT_N_SAMPLES = 1000


class MLE(AbstractModelBaseClass):
    """
    Maximum Likelihood Estimate (MLE) model built around the scipy.stats package.
    """

    def __init__(
            self,
            extremes: pd.Series,
            distribution: str,
            distribution_kwargs: dict = None,
            **kwargs
    ) -> None:
        super().__init__(
            extremes=extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            **kwargs
        )
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        logger.info('initializing the fit parameter hash')
        self.hashed_fit_parameters = []

    @property
    def name(self) -> str:
        return 'MLE'

    def fit(self) -> None:
        self.fit_parameters = self.distribution.mle_parameters
        self.trace = None

    def _encode_kwargs(
            self,
            kwargs: dict
    ) -> str:
        n_samples = kwargs.get('n_samples', DEFAULT_N_SAMPLES)
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
        return_value = getattr(self, self.extreme_value_function)(exceedance_probability)

        if alpha is None:
            if 'n_samples' in kwargs:
                kwargs.pop('n_samples')
                warnings.warn(message='n_samples is not used when alpha is None')
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            logger.debug('returning confidence interval as None for alpha=None')
            confidence_interval = (None, None)
        else:
            n_samples = kwargs.pop('n_samples', DEFAULT_N_SAMPLES)
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            if len(self.hashed_fit_parameters) < n_samples:
                logger.debug(
                    f'putting {n_samples-len(self.hashed_fit_parameters):d} additional fit parameters into the hash'
                )
                for _ in range(n_samples - len(self.hashed_fit_parameters)):
                    sample = np.random.choice(a=self.extremes.values, size=len(self.extremes), replace=True)
                    sample_fit_parameters = self.distribution.distribution.fit(
                        data=sample,
                        **self.distribution.fixed_parameters
                    )
                    self.hashed_fit_parameters.append(sample_fit_parameters)

            logger.debug(
                'calculating return values from hashed fit parameters to be used for confidence interval estimation'
            )
            extreme_value_function = getattr(self.distribution.distribution, self.extreme_value_function)
            rv_sample = np.array(
                [
                    extreme_value_function(exceedance_probability, *self.hashed_fit_parameters[i])
                    for i in range(n_samples)
                ]
            )

            logger.debug('calculating confidence interval')
            confidence_interval = tuple(
                np.quantile(
                    a=rv_sample,
                    q=[(1-alpha)/2, (1+alpha)/2]
                )
            )
        return return_value, confidence_interval
