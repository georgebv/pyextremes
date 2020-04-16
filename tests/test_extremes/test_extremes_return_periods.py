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
import os

import numpy as np
import pandas as pd
import pytest

from pyextremes.extremes import get_extremes, get_return_periods

# Set up logging
logger = logging.getLogger('pyextremes.extremes')
logger.setLevel(logging.CRITICAL)

test_data = pd.read_csv(
    os.sep.join([*os.path.realpath(__file__).split(os.sep)[:-2], 'data', 'battery_wl.csv']),
    index_col=0, parse_dates=True, squeeze=True
)


def test_get_return_periods():
    extremes = get_extremes(
        method='BM',
        ts=test_data,
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
            period_size='1Y',
            plotting_position='weibull',
        )

    # Test automatic block_size type
    extremes_30d = get_extremes(
        method='BM',
        ts=test_data,
        extremes_type='high',
        block_size='30D',
        errors='ignore'
    )
    return_periods_automatic = get_return_periods(
        ts=test_data,
        extremes=extremes_30d,
        extremes_method='BM',
        extremes_type='high',
        block_size=None,
        period_size='1Y',
        plotting_position='weibull',
    )
    return_periods = get_return_periods(
        ts=test_data,
        extremes=extremes_30d,
        extremes_method='BM',
        extremes_type='high',
        block_size=None,
        period_size='1Y',
        plotting_position='weibull',
    )
    assert np.abs(
        np.diff(
            [
                return_periods_automatic.loc[:, 'return period'].values.max(),
                return_periods.loc[:, 'return period'].values.max()
            ]
        )[0]
    ) <= 1

    # Test bad period_size type
    with pytest.raises(TypeError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BM',
            extremes_type='high',
            block_size='1Y',
            period_size=1,
            plotting_position='weibull',
        )

    # Test bad extremes_method
    with pytest.raises(ValueError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BAD EXTREMES METHOD',
            extremes_type='high',
            block_size='1Y',
            period_size='1Y',
            plotting_position='weibull',
        )

    # Test bad extremes_type
    with pytest.raises(ValueError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BM',
            extremes_type='BAD EXTREMES TYPE',
            plotting_position='weibull',
            block_size='1Y'
        )

    # Test bad plotting_position
    with pytest.raises(ValueError):
        get_return_periods(
            ts=test_data,
            extremes=extremes,
            extremes_method='BM',
            extremes_type='high',
            plotting_position='BAD PLOTTING POSITION',
            block_size='1Y'
        )


def test_get_return_periods_bm():
    # Test for BM
    for extremes_type in ['high', 'low']:
        extremes = get_extremes(
            method='BM',
            ts=test_data,
            block_size='1Y',
            extremes_type=extremes_type,
            errors='ignore'
        )
        for plotting_position in [
            'ecdf', 'hazen', 'weibull', 'tukey', 'blom', 'median', 'cunnane', 'gringorten', 'beard'
        ]:
            return_periods = get_return_periods(
                ts=test_data,
                extremes=extremes,
                extremes_method='BM',
                extremes_type=extremes_type,
                plotting_position=plotting_position,
                block_size='1Y',
                period_size='1Y'
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


def test_get_return_period_pot():
    # Test for POT
    for extremes_type in ['high', 'low']:
        extremes = get_extremes(
            method='POT',
            ts=test_data,
            threshold={
                'high': 1.35,
                'low': -1.65
            }[extremes_type],
            r='24H'
        )
        for plotting_position in [
            'ecdf', 'hazen', 'weibull', 'tukey', 'blom', 'median', 'cunnane', 'gringorten', 'beard'
        ]:
            return_periods = get_return_periods(
                ts=test_data,
                extremes=extremes,
                extremes_method='POT',
                extremes_type=extremes_type,
                plotting_position=plotting_position,
                period_size='1Y'
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
