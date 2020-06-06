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

from pyextremes.models.distribution import Distribution


def test_distribution_errors():
    # Test bad distribution type
    with pytest.raises(TypeError):
        Distribution(
            extremes=pd.Series([1, 2, 3]),
            distribution=1
        )

    # Test discrete distribution
    with pytest.raises(ValueError):
        Distribution(
            extremes=pd.Series([1, 2, 3]),
            distribution='poisson'
        )

    # Test bad kwargs
    with pytest.raises(TypeError):
        Distribution(
            extremes=pd.Series([1, 2, 3]),
            distribution='expon',
            fc=0
        )


@pytest.mark.parametrize(
    'distribution_name, theta, kwargs, scipy_parameters',
    [
        ('genextreme', (0.5, 10, 2), {}, (0.5, 10, 2)),
        ('gumbel_r', (10, 2), {}, (10, 2)),
        ('genpareto', (0.5, 0, 2), {}, (0.5, 0, 2)),
        ('genpareto', (0.5, 2), {'floc': 0}, (0.5, 0, 2)),
        ('expon', (0, 2,), {}, (0, 2)),
        ('expon', (2, ), {'floc': 0}, (0, 2))
    ]
)
def test_distribution(distribution_name, theta, kwargs, scipy_parameters):
    scipy_distribution = getattr(scipy.stats, distribution_name)

    # Test distribution fit
    distribution = Distribution(
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=100, freq='1H'),
            data=scipy_distribution.rvs(*scipy_parameters, size=100)
        ),
        distribution=distribution_name,
        **kwargs
    )
    assert len(distribution.distribution_parameters) == len(scipy_parameters)
    assert distribution.fixed_parameters == kwargs
    assert len(distribution._fixed_parameters) == len(kwargs)
    assert len(distribution.free_parameters) == (len(scipy_parameters) - len(kwargs))
    assert len(distribution.mle_parameters) == distribution.number_of_parameters
    assert distribution.name == distribution_name

    # Test log-probability
    assert np.isclose(
        distribution.log_probability(theta=theta),
        sum(scipy_distribution.logpdf(distribution.extremes.values, *scipy_parameters))
    )

    # Test initial state
    initial_state = distribution.get_initial_state(n_walkers=1000)
    assert initial_state.shape == (1000, distribution.number_of_parameters)
    assert np.allclose(initial_state.mean(axis=0), list(distribution.mle_parameters.values()), atol=0.1)

    # Test free2full parameters
    assert len(distribution.free2full_parameters(free_parameters=distribution.mle_parameters)) == len(scipy_parameters)
    assert len(distribution.free2full_parameters(free_parameters=theta)) == len(scipy_parameters)
    assert distribution.free2full_parameters(free_parameters=[theta]*5).shape == (5, len(scipy_parameters))

    # Test _get_prop
    for prop in ['pdf', 'cdf', 'ppf', 'isf']:
        free_parameters = {key: theta[i] for i, key in enumerate(distribution.free_parameters)}
        assert np.isclose(
            distribution.get_prop(prop=prop, x=0.1, free_parameters=free_parameters),
            getattr(scipy_distribution, prop)(0.1, *scipy_parameters)
        )
        assert np.allclose(
            distribution.get_prop(prop=prop, x=[0.1, 0.2], free_parameters=free_parameters),
            getattr(scipy_distribution, prop)([0.1, 0.2], *scipy_parameters)
        )
        assert np.allclose(
            distribution.get_prop(prop=prop, x=0.1, free_parameters=[theta, theta]),
            getattr(scipy_distribution, prop)(0.1, *np.transpose([scipy_parameters, scipy_parameters]))
        )
        assert np.allclose(
            distribution.get_prop(prop=prop, x=[0.1, 0.2], free_parameters=[theta, theta]),
            np.transpose(
                getattr(scipy_distribution, prop)(
                    np.transpose([0.1, 0.2]),
                    *np.transpose(
                        [[scipy_parameters, scipy_parameters], [scipy_parameters, scipy_parameters]]
                    )
                )
            )
        )
