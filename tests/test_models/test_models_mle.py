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
    'distribution_name, parameters, scipy_parameters',
    [
        ('genextreme', (0.5, 10, 2), (0.5, 10, 2)),
        ('genpareto', (0.5, 2), (0.5, 0, 2))
    ]
)
def test_mle_fit(distribution_name, parameters, scipy_parameters):
    model = get_model(
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=60, freq='1H'),
            data=getattr(scipy.stats, distribution_name).rvs(*scipy_parameters, size=60)
        ),
        model='MLE',
        distribution=distribution_name
    )
    assert isinstance(model.fit_parameters, dict)
    assert len(model.fit_parameters) == len(scipy_parameters)


def test_mle():
    # Test fit
    parameters = (0.5, 10, 2)
    model = get_model(
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=60, freq='1H'),
            data=scipy.stats.genextreme.rvs(*parameters, size=60)
        ),
        model='MLE',
        distribution='genextreme'
    )
    assert isinstance(model.fit_parameters, dict)
    assert len(model.fit_parameters) == len(parameters)

    # Test bad exceedance_probability type
    with pytest.raises(TypeError):
        model.get_return_value(exceedance_probability='0.1', alpha=0.95, n_samples=40)

    # Test bad n_samples values
    with pytest.raises(TypeError):
        model.get_return_value(exceedance_probability=0.1, alpha=0.95, n_samples=1.1)
    with pytest.raises(ValueError):
        model.get_return_value(exceedance_probability=0.1, alpha=0.95, n_samples=-1)

    # Test return value
    return_value = model.get_return_value(exceedance_probability=0.1, alpha=0.95, n_samples=40)
    assert len(return_value) == 3
    assert return_value[1] < return_value[0] < return_value[2]

    # Test hash
    hashed_rv = model.get_return_value(exceedance_probability=0.1, alpha=0.95, n_samples=40)
    assert len(model.hashed_return_values) == 1
    assert return_value == hashed_rv

    # Test alpha being None
    return_value = model.get_return_value(exceedance_probability=0.1, alpha=None)
    assert return_value[1:] == (None, None)

    # Test update hashed values
    model.get_return_value(exceedance_probability=0.1, alpha=0.95, n_samples=20)
    assert len(model.hashed_return_values) == 1
    assert len(model.hashed_return_values['0.100000']) == 3
    assert len(model.hashed_return_values['0.100000']['0.950000']) == 2
    model.get_return_value(exceedance_probability=0.1, alpha=0.5, n_samples=20)
    assert len(model.hashed_return_values) == 1
    assert len(model.hashed_return_values['0.100000']) == 4
    assert len(model.hashed_return_values['0.100000']['0.500000']) == 1

    # Get multiple return values
    return_values = model.get_return_value(exceedance_probability=np.arange(0.9, 0, -0.1), alpha=0.5, n_samples=10)
    assert len(return_values) == 3
    assert len(model.hashed_return_values) == 9

    # Test pdf and cdf
    for prop in ['pdf', 'cdf']:
        assert np.isclose(
            getattr(model, prop)(0.1),
            getattr(scipy.stats.genextreme, prop)(0.1, *model.fit_parameters)
        )

    # Test hashed fit parameters
    assert len(model.hashed_fit_parameters) == 40
    assert np.all([len(sample) == 3 for sample in model.hashed_fit_parameters])
