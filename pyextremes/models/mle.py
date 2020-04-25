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
import pandas as pd
import scipy.stats

from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)


class MLE(AbstractModelBaseClass):

    def __init__(
            self,
            extremes: pd.Series,
            distribution: str
    ) -> None:
        super().__init__()

        self.extremes = extremes

        logger.info('getting extreme value distribution')
        if isinstance(distribution, str):
            try:
                logger.info(f'getting distribution object with name \'{distribution}\' from scipy.stats')
                self.distribution = getattr(scipy.stats, distribution)
            except AttributeError:
                raise ValueError(f'\'{distribution}\' is not a valid \'distribution\' value')
        else:
            raise TypeError(f'invalid type in {type(distribution)} for the \'distribution\' argument')

        logger.info('fitting the distribution to extremes')
        self.fit_parameters = self.fit(extremes=extremes)

    def fit(
            self,
            extremes: pd.Series,
            **kwargs
    ) -> tuple:
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
        distributions = ['genpareto', 'genextreme']
        if self.distribution.name == 'genpareto':
            return self.distribution.fit(extremes, floc=0)
        elif self.distribution.name == 'genextreme':
            return self.distribution.fit(extremes)
        else:
            raise NotImplementedError(
                f'\'{self.distribution.name}\' distribution is not implemented for the \'MLE\' model\n'
                f'Available \'MLE\' distributions: {", ".join(distributions)}'
            )

    def retrieve_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs: dict
    ) -> tuple:
        n_samples = kwargs.pop('n_samples', 1000)
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        logger.debug('checking if a result for given parameters has been previously hashed')
        condition_1 = exceedance_probability not in self.hashed_return_values
        if condition_1:
            condition_2 = True
        else:
            condition_2 = alpha not in self.hashed_return_values[exceedance_probability]
        if condition_2:
            condition_3 = True
        else:
            condition_3 = n_samples not in self.hashed_return_values[exceedance_probability][alpha]
        if condition_1 or condition_2 or condition_3:
            logger.debug('full entry not hashed, calculating return value')
            rv = self.__get_return_value(
                exceedance_probability=exceedance_probability,
                alpha=alpha,
                n_samples=n_samples
            )
            if condition_1:
                logger.debug(f'creating a new entry for exceedance_probability {exceedance_probability}')
                self.hashed_return_values[exceedance_probability] = {
                    'return value': rv[0],
                    alpha: {
                        n_samples: rv[1]
                    }
                }
            elif condition_2:
                logger.debug(f'updating entry for exceedance_probability {exceedance_probability} with alpha {alpha}')
                self.hashed_return_values[exceedance_probability][alpha] = {
                    n_samples: rv[1]
                }
            elif condition_3:
                logger.debug(
                    f'updating entry for exceedance_probability {exceedance_probability} and alpha {alpha} '
                    f'with n_samples {n_samples}'
                )
                self.hashed_return_values[exceedance_probability][alpha][n_samples] = rv[1]

        logger.debug('retrieving hashed entry and returning results')
        hashed_entry = self.hashed_return_values[exceedance_probability]
        return (hashed_entry['return value'], *hashed_entry[alpha][n_samples])

    def __get_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            n_samples: int
    ) -> tuple:
        if not isinstance(n_samples, int):
            raise TypeError(f'invalid type in {type(n_samples)} for the \'n_samples\' argument')
        if n_samples <= 0:
            raise ValueError(f'\'{n_samples}\' is not a valid \'n_samples\' value, it must be a positive integer')

        return_value = self.distribution.isf(exceedance_probability, *self.fit_parameters)
        if alpha is None:
            confidence_interval = (None, None)
        else:
            def draw_sample(sample_size):
                sample = np.random.choice(a=self.extremes.values, size=sample_size, replace=True)
                sample_fit_parameters = self.fit(extremes=sample)
                return self.distribution.isf(exceedance_probability, *sample_fit_parameters)
            vectorized_draw_sample = np.vectorize(draw_sample)
            rv_sample = vectorized_draw_sample(scipy.stats.poisson.rvs(len(self.extremes), size=n_samples))

            confidence_interval = tuple(
                np.quantile(
                    a=rv_sample,
                    q=[(1-alpha)/2, (1+alpha)/2]
                )
            )
        return return_value, confidence_interval
