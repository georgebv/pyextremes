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

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from pyextremes.models.emcee.distributions import get_distribution


def test_get_distribution():
    # Test bad distribution
    with pytest.raises(ValueError):
        get_distribution(
            distribution='BAD DISTRIBUTION',
            extremes=pd.Series([1, 2, 3])
        )


def test_genextreme():
    # Test distribution fit
    parameters = (0.5, 10, 2)
    distribution = get_distribution(
        distribution='genextreme',
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=100, freq='1H'),
            data=scipy.stats.genextreme.rvs(*parameters, size=100)
        )
    )
    assert len(distribution.mle_parameters) == distribution.number_of_parameters

    # Test log-likelihood
    assert np.isclose(
        distribution.log_likelihood(theta=parameters),
        sum(scipy.stats.genextreme.logpdf(distribution.extremes.values, *parameters))
    )

    # Test log-probability
    assert np.greater_equal(
        distribution.log_probability(theta=distribution.mle_parameters),
        distribution.log_probability(theta=parameters)
    )

    # Test initial state
    initial_state = distribution.get_initial_state(n_walkers=100)
    assert initial_state.shape == (100, distribution.number_of_parameters)
    assert np.allclose(initial_state.mean(axis=0), distribution.mle_parameters, atol=0.1)

    # Test isf
    assert np.isclose(
        distribution.isf(q=0.1, parameters=parameters),
        scipy.stats.genextreme.isf(0.1, *parameters)
    )


def test_genpareto():
    # Test distribution fit
    parameters = (0.5, 2)
    distribution = get_distribution(
        distribution='genpareto',
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=100, freq='1H'),
            data=scipy.stats.genpareto.rvs(c=parameters[0], loc=0, scale=parameters[1], size=100)
        )
    )
    assert len(distribution.mle_parameters) == distribution.number_of_parameters

    # Test log-likelihood
    assert np.isclose(
        distribution.log_likelihood(theta=parameters),
        sum(scipy.stats.genpareto.logpdf(distribution.extremes.values, c=parameters[0], loc=0, scale=parameters[1]))
    )

    # Test log-probability
    assert np.greater_equal(
        distribution.log_probability(theta=distribution.mle_parameters),
        distribution.log_probability(theta=parameters)
    )

    # Test initial state
    initial_state = distribution.get_initial_state(n_walkers=100)
    assert initial_state.shape == (100, distribution.number_of_parameters)
    assert np.allclose(initial_state.mean(axis=0), distribution.mle_parameters, atol=0.1)

    # Test isf
    assert np.isclose(
        distribution.isf(q=0.1, parameters=parameters),
        scipy.stats.genpareto.isf(0.1, c=parameters[0], loc=0, scale=parameters[1])
    )


def test_gumbel_r():
    # Test distribution fit
    parameters = (10, 2)
    distribution = get_distribution(
        distribution='gumbel_r',
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=100, freq='1H'),
            data=scipy.stats.gumbel_r.rvs(*parameters, size=100)
        )
    )
    assert len(distribution.mle_parameters) == distribution.number_of_parameters

    # Test log-likelihood
    assert np.isclose(
        distribution.log_likelihood(theta=parameters),
        sum(scipy.stats.gumbel_r.logpdf(distribution.extremes.values, *parameters))
    )

    # Test log-probability
    assert np.greater_equal(
        distribution.log_probability(theta=distribution.mle_parameters),
        distribution.log_probability(theta=parameters)
    )

    # Test initial state
    initial_state = distribution.get_initial_state(n_walkers=100)
    assert initial_state.shape == (100, distribution.number_of_parameters)
    assert np.allclose(initial_state.mean(axis=0), distribution.mle_parameters, atol=0.1)

    # Test isf
    assert np.isclose(
        distribution.isf(q=0.1, parameters=parameters),
        scipy.stats.gumbel_r.isf(0.1, *parameters)
    )


def test_expon():
    # Test distribution fit
    parameters = (2, )
    distribution = get_distribution(
        distribution='expon',
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=100, freq='1H'),
            data=scipy.stats.expon.rvs(loc=0, scale=parameters[0], size=100)
        )
    )
    assert len(distribution.mle_parameters) == distribution.number_of_parameters

    # Test log-likelihood
    assert np.isclose(
        distribution.log_likelihood(theta=parameters),
        sum(scipy.stats.expon.logpdf(distribution.extremes.values, loc=0, scale=parameters[0]))
    )

    # Test log-probability
    assert np.greater_equal(
        distribution.log_probability(theta=distribution.mle_parameters),
        distribution.log_probability(theta=parameters)
    )

    # Test initial state
    initial_state = distribution.get_initial_state(n_walkers=100)
    assert initial_state.shape == (100, distribution.number_of_parameters)
    assert np.allclose(initial_state.mean(axis=0), distribution.mle_parameters, atol=0.1)

    # Test isf
    assert np.isclose(
        distribution.isf(q=0.1, parameters=parameters),
        scipy.stats.expon.isf(0.1, loc=0, scale=parameters[0])
    )
