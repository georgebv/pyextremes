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


def test_emcee():
    # Test fit
    parameters = (0.5, 10, 2)
    model = get_model(
        extremes=pd.Series(
            index=pd.date_range(start='2000-01-01', periods=100, freq='1H'),
            data=scipy.stats.genextreme.rvs(*parameters, size=100)
        ),
        model='Emcee',
        distribution='genextreme',
        n_walkers=20,
        n_samples=100
    )
    assert isinstance(model.fit_parameters, dict)
    assert len(model.fit_parameters['map']) == len(parameters)
    assert model.fit_parameters['trace'].shape == (20, 100, 3)

    # Test bad exceedance_probability type
    with pytest.raises(TypeError):
        model.get_return_value(exceedance_probability='0.1', alpha=0.95, burn_in=20)

    # Test bad burn_in values
    with pytest.raises(KeyError):
        model.get_return_value(exceedance_probability=0.1, alpha=0.95)
    with pytest.raises(TypeError):
        model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=1.1)
    with pytest.raises(ValueError):
        model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=-1)
    with pytest.raises(ValueError):
        model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=100)

    # Test return value
    return_value = model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=20)
    assert len(return_value) == 3
    assert return_value[1] < return_value[0] < return_value[2]

    # Test hash
    hashed_rv = model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=20)
    assert len(model.hashed_return_values) == 1
    assert return_value == hashed_rv

    # Test alpha being None
    return_value = model.get_return_value(exceedance_probability=0.1, alpha=None)
    assert return_value[1:] == (None, None)

    # Test update hashed values
    model.get_return_value(exceedance_probability=0.1, alpha=0.95, burn_in=40)
    assert len(model.hashed_return_values) == 1
    assert len(model.hashed_return_values['0.100000']) == 3
    assert len(model.hashed_return_values['0.100000']['0.950000']) == 2
    model.get_return_value(exceedance_probability=0.1, alpha=0.5, burn_in=20)
    assert len(model.hashed_return_values) == 1
    assert len(model.hashed_return_values['0.100000']) == 4
    assert len(model.hashed_return_values['0.100000']['0.500000']) == 1

    # Get multiple return values
    return_values = model.get_return_value(exceedance_probability=np.arange(0.9, 0, -0.1), alpha=0.5, burn_in=20)
    assert len(return_values) == 3
    assert len(model.hashed_return_values) == 9
