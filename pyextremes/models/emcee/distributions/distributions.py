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

import pandas as pd

from pyextremes.models.emcee.distributions.distribution_base import AbstractEmceeDistributionBaseClass
from pyextremes.models.emcee.distributions.expon import Expon
from pyextremes.models.emcee.distributions.genextreme import Genextreme
from pyextremes.models.emcee.distributions.genpareto import Genpareto
from pyextremes.models.emcee.distributions.gumbel_r import GumbelR

logger = logging.getLogger(__name__)

distributions = {
    'genextreme': Genextreme,
    'genpareto': Genpareto,
    'gumbel_r': GumbelR,
    'expon': Expon
}


def get_distribution(
        distribution: str,
        extremes: pd.Series,
) -> AbstractEmceeDistributionBaseClass:
    """
    Get a distribution object to be used within the Emcee fitting model.

    Parameters
    ----------
    distribution : str
        Distribution name compatible with scipy.stats.
    extremes : pandas.Series
        Time series of transformed extreme events.

    Returns
    -------
    Emcee-compatible distribution object
    """

    logger.info(f'fetching \'{distribution}\' distribution')
    try:
        distribution = distributions[distribution]
    except KeyError:
        raise ValueError(
            f'\'{distribution}\' distribution is not available for the \'Emcee\' model\n'
            f'Available \'Emcee\' distributions: {", ".join(distributions.keys())}'
        )
    logger.info('creating distribution object')
    return distribution(extremes=extremes)
