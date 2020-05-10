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
import scipy.stats

import pytest
from pyextremes.models import get_model


@pytest.mark.parametrize(
    'distribution_name, theta, distribution_kwargs, scipy_parameters',
    [
        ('genextreme', (0.5, 10, 2), {}, (0.5, 10, 2)),
        ('gumbel_r', (10, 2), {}, (10, 2)),
        ('genpareto', (0.5, 0, 2), {}, (0.5, 0, 2)),
        ('genpareto', (0.5, 2), {'floc': 0}, (0.5, 0, 2)),
        ('expon', (0, 2,), {}, (0, 2)),
        ('expon', (2, ), {'floc': 0}, (0, 2))
    ]
)
def test_emcee(distribution_name, theta, distribution_kwargs, scipy_parameters):
    scipy_distribution = getattr(scipy.stats, distribution_name)

    model = get_model(
        model='Emcee',
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=100, freq='1H'),
            data=scipy_distribution.rvs(*scipy_parameters, size=100)
        ),
        distribution=distribution_name,
        distribution_kwargs=distribution_kwargs,
        n_walkers=20,
        n_samples=100
    )

    # Test fit
    assert len(model.fit_parameters) == (len(scipy_parameters) - len(distribution_kwargs))
    assert model.trace.shape == (20, 100, len(theta))
    assert model.hashed_return_values == {}

    # Test model name
    assert model.name == 'Emcee'

    # Test loglikelihood
    assert np.isclose(
        model.loglikelihood,
        sum(
            scipy_distribution.logpdf(
                model.extremes.values,
                **model.fit_parameters,
                **model.distribution._fixed_parameters
            )
        )
    )

    # Test AIC
    k = model.distribution.number_of_parameters
    n = len(model.extremes)
    loglikelihood = sum(
        scipy_distribution.logpdf(
            model.extremes.values,
            **model.fit_parameters,
            **model.distribution._fixed_parameters
        )
    )
    aic = 2 * k - 2 * loglikelihood
    correction = (2 * k ** 2 + 2 * k) / (n - k - 1)
    assert np.isclose(
        model.AIC,
        aic + correction
    )

    # Test encode_kwargs
    with pytest.raises(TypeError):
        model._encode_kwargs({'burn_in': 1.01})
    with pytest.raises(ValueError):
        model._encode_kwargs({'burn_in': -5})
    with pytest.raises(ValueError):
        model._encode_kwargs({'burn_in': 100})
    for burn_in in [1, 10, 95]:
        assert model._encode_kwargs({'burn_in': burn_in}) == f'{burn_in:d}'

    # Test return value
    return_value = model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=50)
    assert np.isclose(
        return_value[0],
        scipy_distribution.isf(0.1, **model.fit_parameters, **model.distribution._fixed_parameters),
        rtol=0.1
    )
    assert len(return_value) == 3
    assert return_value[1] < return_value[0] < return_value[2]

    # Test hash
    hashed_rv = model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=50)
    assert len(model.hashed_return_values) == 1
    assert np.allclose(return_value, hashed_rv)

    # Test alpha being None
    return_value = model.get_return_value(exceedance_probability=0.1, alpha=None)
    assert return_value[1:] == (None, None)

    # Test update hashed values
    model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=20)
    assert len(model.hashed_return_values) == 1
    assert len(model.hashed_return_values['0.100000']) == 3
    assert len(model.hashed_return_values['0.100000']['0.950000']) == 2
    model.get_return_value(exceedance_probability=0.1, alpha=0.5, burn_in=20)
    assert len(model.hashed_return_values) == 1
    assert len(model.hashed_return_values['0.100000']) == 4
    assert len(model.hashed_return_values['0.100000']['0.500000']) == 1

    # Get multiple return values
    return_values = model.get_return_value(exceedance_probability=np.arange(0.9, 0, -0.1), alpha=0.5, burn_in=10)
    assert len(return_values) == 3
    assert len(model.hashed_return_values) == 9

    # Test properties
    for prop in ['pdf', 'cdf', 'ppf', 'isf', 'logpdf']:
        assert np.isclose(
            getattr(model, prop)(0.1),
            getattr(scipy_distribution, prop)(0.1, **model.fit_parameters, **model.distribution._fixed_parameters)
        )
