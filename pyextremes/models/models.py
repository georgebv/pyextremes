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

from pyextremes.models.mle import MLE
from pyextremes.models.pymc import PyMC

logger = logging.getLogger(__name__)


def get_model(
        model: str
) -> type:
    """
    Get an extreme value model.

    Parameters
    ----------
    model : str
        Name of an extreme value distribution fitting model.
        Supported names:
            MLE - Maximum Likelihood Estimate model (based on scipy)
            PyMC - PyMC3 Hamiltonian Monte Carlo model

    Returns
    -------
    fitting_model : class
        An extreme value model.
    """

    logger.info(f'calling get_fitting_model with model={model}')
    if model == 'MLE':
        return MLE
    elif model == 'PyMC':
        return PyMC
    else:
        raise ValueError(f'\'{model}\' is not a valid \'model\' value')
