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

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from pyextremes.extremes import get_extremes, get_return_periods

test_data_folder = pathlib.Path(os.path.realpath(__file__)).parent.parent / 'data'
test_data = pd.read_csv(test_data_folder/'battery_wl.csv', index_col=0, parse_dates=True, squeeze=True)


def test_get_return_periods():
    extremes = get_extremes(
        ts=test_data,
        method='BM',
        extremes_type='high',
        block_size='1Y',
        errors='ignore'
    )

    # Test bad block_size type
    with pytest.raises(TypeError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BM',
            extremes_type='high',
            block_size=1,
            return_period_size='1Y',
            plotting_position='weibull'
        )

    # Test block_size for POT
    with pytest.raises(ValueError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='POT',
            extremes_type='high',
            block_size='1Y',
            return_period_size='1Y',
            plotting_position='weibull'
        )

    # Test automatic block_size type
    return_periods_automatic = get_return_periods(
        ts=test_data,
        extremes=extremes,
        extremes_method='BM',
        extremes_type='high',
        block_size=None,
        return_period_size='1Y',
        plotting_position='weibull'
    )
    return_periods = get_return_periods(
        ts=test_data,
        extremes=extremes,
        extremes_method='BM',
        extremes_type='high',
        block_size='1Y',
        return_period_size='1Y',
        plotting_position='weibull'
    )
    assert np.abs(
        return_periods_automatic.loc[:, 'return period'].values.max() -
        return_periods.loc[:, 'return period'].values.max()
    ) <= 1

    # Test bad return_period_size type
    with pytest.raises(TypeError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BM',
            extremes_type='high',
            block_size='1Y',
            return_period_size=1,
            plotting_position='weibull'
        )

    # Test bad extremes_method
    with pytest.raises(ValueError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BAD EXTREMES METHOD',
            extremes_type='high',
            block_size=None,
            return_period_size='1Y',
            plotting_position='weibull'
        )

    # Test bad extremes_type
    with pytest.raises(ValueError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BM',
            extremes_type='BAD EXTREMES TYPE',
            block_size='1Y',
            return_period_size='1Y',
            plotting_position='weibull'
        )

    # Test bad plotting_position
    with pytest.raises(ValueError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BM',
            extremes_type='high',
            block_size='1Y',
            return_period_size='1Y',
            plotting_position='BAD PLOTTING POSITION'
        )


def test_extremes_method_bm():
    for extremes_type in ['high', 'low']:
        extremes = pd.read_csv(
            test_data_folder/f'extremes_bm_{extremes_type}.csv',
            index_col=0, parse_dates=True, squeeze=True
        )
        for plotting_position in [
            'ecdf', 'hazen', 'weibull', 'tukey', 'blom', 'median', 'cunnane', 'gringorten', 'beard'
        ]:
            return_periods = get_return_periods(
                ts=test_data,
                extremes=extremes,
                extremes_method='BM',
                extremes_type=extremes_type,
                block_size='1Y',
                return_period_size='1Y',
                plotting_position=plotting_position
            )
            if extremes_type == 'high':
                assert np.argmax(
                    return_periods.loc[:, extremes.name].values
                ) == np.argmax(
                    return_periods.loc[:, 'return period'].values
                )
            else:
                assert np.argmin(
                    return_periods.loc[:, extremes.name].values
                ) == np.argmax(
                    return_periods.loc[:, 'return period'].values
                )


def test_extremes_method_pot():
    for extremes_type in ['high', 'low']:
        extremes = pd.read_csv(
            test_data_folder/f'extremes_pot_{extremes_type}.csv',
            index_col=0, parse_dates=True, squeeze=True
        )
        for plotting_position in [
            'ecdf', 'hazen', 'weibull', 'tukey', 'blom', 'median', 'cunnane', 'gringorten', 'beard'
        ]:
            return_periods = get_return_periods(
                ts=test_data,
                extremes=extremes,
                extremes_method='POT',
                extremes_type=extremes_type,
                block_size=None,
                return_period_size='1Y',
                plotting_position=plotting_position
            )
            if extremes_type == 'high':
                assert np.argmax(
                    return_periods.loc[:, extremes.name].values
                ) == np.argmax(
                    return_periods.loc[:, 'return period'].values
                )
            else:
                assert np.argmin(
                    return_periods.loc[:, extremes.name].values
                ) == np.argmax(
                    return_periods.loc[:, 'return period'].values
                )
