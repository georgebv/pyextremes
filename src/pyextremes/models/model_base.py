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

import abc
import logging
import typing

import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.models.distribution import Distribution

logger = logging.getLogger(__name__)


class AbstractModelBaseClass(abc.ABC):
    """
    Distribution model base class.
    """

    def __init__(
            self,
            extremes: pd.Series,
            distribution: typing.Union[str, scipy.stats.rv_continuous],
            distribution_kwargs: dict = None,
            **kwargs
    ) -> None:
        """
        Initialize the distribution model object.

        Parameters
        ----------
        extremes : pandas.Series
            Time series of extreme events.
        distribution : str or scipy.stats.rv_continuous
            Distribution name compatible with scipy.stats or a subclass of scipy.stats.rv_continuous
            See https://docs.scipy.org/doc/scipy/reference/stats.html for a list of continuous distributions
        distribution_kwargs : dict, optional
            Dictionary with special keyword arguments, passsed to the .fit method of the continuous distribution.
            These keyword arguments represent parameters to be held fixed and must be shape, scale, or location
            parameter names with sufix 'f', e.g. 'fc', 'floc', or 'fscale'. By default no parameters are fixed.
            See documentation of a specific scipy.stats distribution for names of available parameters.
            Examples:
                dict(fc=0) holds shape parameter 'c' at 0 essentially eliminating it as an independent parameter
                    of the distribution, reducting its degree of freedom (number of free parameters) by one.
                dict(floc=0) hold the location parameter 'loc' at 0
                dict(fc=0, floc=10) holds shape and location parameters fixed at 0 and 10 respectively
        kwargs
            Keyword arguments passed to a model .fit method.
            MLE model:
                MLE model takes no additional arguments.
            Emcee model:
                n_walkers : int, optional
                    The number of walkers in the ensemble (default=100).
                n_samples : int, optional
                    The number of steps to run (default=500).
                progress : bool or str, optional
                    If True, a progress bar will be shown as the sampler progresses.
                    If a string, will select a specific tqdm progress bar - most notable is
                    'notebook', which shows a progress bar suitable for Jupyter notebooks.
                    If False, no progress bar will be shown (default=False).
                    This progress bar is a part of the emcee package.
        """

        self.extremes = extremes

        logger.info('fetching extreme value distribution')
        distribution_kwargs = distribution_kwargs or {}
        self.distribution = Distribution(extremes=self.extremes, distribution=distribution, **distribution_kwargs)

        logger.info('fitting the distribution to extremes')
        self.fit_parameters = None
        self.trace = None
        self.fit(**kwargs)

        logger.info('initializing the return value hash')
        self.hashed_return_values = {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns model name
        """
        pass

    @abc.abstractmethod
    def fit(self, **kwargs) -> None:
        """
        Sets values for self.fit_parameters and self.trace (the latter only for MCMC-like models).
        self.fit_parameters is a dictionary with {parameter_name: value}, e.g. {'c': 0.1, 'loc': -7, 'scale': 0.3}
        self.trace is a numpy.ndarray with shape of (n_walkers, n_samples, n_free_parameters)
        """
        pass

    @property
    def loglikelihood(self) -> float:
        return sum(self.logpdf(x=self.extremes.values))

    # noinspection PyPep8Naming
    @property
    def AIC(self) -> float:
        """
        Returns corrected Akaike Information Criterion (AIC) of the model.
        See https://en.wikipedia.org/wiki/Akaike_information_criterion
        """

        k = self.distribution.number_of_parameters
        n = len(self.extremes)
        aic = 2 * k - 2 * self.loglikelihood
        correction = (2 * k ** 2 + 2 * k) / (n - k - 1)
        return aic + correction

    @abc.abstractmethod
    def _encode_kwargs(self, kwargs: dict) -> str:
        """
        Convert kwargs to a string, which is used as a "return value" and "confidence interval" hash keys.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keyword arguments passed to a model self._get_return_value method.

        Returns
        -------
        encoded_kwargs : str
            String with encoded kwargs.
        """
        pass

    def get_return_value(
            self,
            exceedance_probability: typing.Union[float, tuple, list, np.ndarray],
            alpha: float = None,
            **kwargs
    ) -> tuple:
        """
        Get return value and confidence interval for a given exceedance probability.

        Parameters
        ----------
        exceedance_probability : float or array-like
            Exceedance probability or array of exceedance probabilities.
            Each exceedance probability must fall in range [0, 1).
        alpha : float, optional
            Width of confidence interval (default=None).
            Must fall in range (0, 1).
            If None, return None for upper and lower confidence interval bounds.
        kwargs
            Keyword arguments passed to a model self._get_return_value method.
            If alpha is None, keyword arguments are ignored (error still raised for unrecognized arguments).
            MLE model:
                n_samples : int, optional
                    Number of bootstrap samples used to estimate confidence interval bounds (default=100).
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        return_value : float or array-like
            Return value(s).
        lower_ci_bound : float or array-like
            Lower confidence interval bound(s).
        upper_ci_bount : float or array-like
            Upper confidence interval bound(s).
        """

        if isinstance(exceedance_probability, (tuple, list, np.ndarray)):
            logger.info('getting an array of return values')
            return tuple(
                np.transpose(
                    [
                        self._retrieve_return_value(exceedance_probability=ep, alpha=alpha, **kwargs)
                        for ep in exceedance_probability
                    ]
                ).astype(float)
            )
        elif isinstance(exceedance_probability, (int, float, np.number)):
            logger.info('getting a single return value')
            return self._retrieve_return_value(exceedance_probability=exceedance_probability, alpha=alpha, **kwargs)
        else:
            raise TypeError(
                f'invalid type in {type(exceedance_probability)} for the \'exceedance_probability\' argument'
            )

    def _retrieve_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs
    ) -> tuple:
        """
        Retrieve return value and confidence interval from hashed results dictionary.
        If the return value has not been previously hashed, calculate it, add it to hash, and return.

        Parameters
        ----------
        exceedance_probability : float
            Exceedance probability [0, 1).
        alpha : float
            Width of confidence interval, from 0 to 1 (default=None).
            If None, return None for upper and lower confidence interval bounds.
        kwargs
            Keyword arguments passed to a model ._get_return_value method.
            If alpha is None, keyword arguments are ignored (error still raised for unrecognized arguments).

        Returns
        -------
        return_value : float
        lower_confidence_interval_bound : float
        upper_confidence_interval_bound : float
        """

        logger.debug('encoding exceedance_probability')
        encoded_exceedance_probability = f'{exceedance_probability:.6f}'

        if alpha is None:
            logger.debug('alpha is None - setting kwargs and alpha to None')
            encoded_kwargs = 'None'
            encoded_alpha = 'None'
        else:
            logger.debug('encoding alpha and kwargs')
            encoded_kwargs = self._encode_kwargs(kwargs=kwargs)
            encoded_alpha = f'{alpha:.6f}'

        try:
            logger.debug(
                f'trying to retrieve result from hash for exceedance_probability {encoded_exceedance_probability}, '
                f'alpha {encoded_alpha}, and kwargs {encoded_kwargs}'
            )
            hashed_entry = self.hashed_return_values[encoded_exceedance_probability]
            return_value = hashed_entry['return value']
            confidence_interval = hashed_entry[encoded_alpha][encoded_kwargs]
            logger.debug('successfully retrieved result from hash - returning values')
            return (return_value, *confidence_interval)
        except KeyError:
            logger.debug(
                f'calculating new results for exceedance_probability {encoded_exceedance_probability}, '
                f'alpha {encoded_alpha}, and kwargs {encoded_kwargs}'
            )
            rv = self._get_return_value(exceedance_probability=exceedance_probability, alpha=alpha, **kwargs)

            if encoded_exceedance_probability in self.hashed_return_values:
                if encoded_alpha in self.hashed_return_values[encoded_exceedance_probability]:
                    logger.debug(
                        f'updating entry for exceedance_probability {encoded_exceedance_probability} '
                        f'and alpha {encoded_alpha} with kwargs {encoded_kwargs}'
                    )
                    self.hashed_return_values[encoded_exceedance_probability][encoded_alpha][encoded_kwargs] = rv[1]
                else:
                    logger.debug(
                        f'updating entry for exceedance_probability {encoded_exceedance_probability} '
                        f'with alpha {encoded_alpha} and kwargs {encoded_kwargs}'
                    )
                    self.hashed_return_values[encoded_exceedance_probability][encoded_alpha] = {encoded_kwargs: rv[1]}
            else:
                logger.debug(
                    f'creating a new entry for exceedance_probability {encoded_exceedance_probability}, '
                    f'alpha {encoded_alpha}, and kwargs {encoded_kwargs}'
                )
                self.hashed_return_values[encoded_exceedance_probability] = {
                    'return value': rv[0],
                    encoded_alpha: {
                        encoded_kwargs: rv[1]
                    }
                }

            logger.debug('hash has been updated - calling the retrieval function again')
            return self._retrieve_return_value(
                exceedance_probability=exceedance_probability,
                alpha=alpha,
                **kwargs
            )

    @abc.abstractmethod
    def _get_return_value(
            self,
            exceedance_probability: float,
            alpha: float,
            **kwargs
    ) -> tuple:
        """
        Calculate return value and confidence interval bounds for given exceedance probability.

        Parameters
        ----------
        exceedance_probability : float
            Exceedance probability [0, 1).
        alpha : float
            Width of confidence interval (0, 1) (default=None).
            If None, return None for upper and lower confidence interval bounds.
        kwargs
            Keyword arguments passed to a model ._get_return_value method.
            If alpha is None, keyword arguments are ignored (error still raised for unrecognized arguments).

        Returns
        -------
        return_value : float
        ci : tuple
            ci_lower : float
            ci_upper : float
        """
        pass

    def get_prop(
            self,
            prop: str,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        logger.debug('calculating and returning property')
        return self.distribution.get_prop(prop=prop, x=x, free_parameters=self.fit_parameters)

    def pdf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.get_prop(prop='pdf', x=x)

    def logpdf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.get_prop(prop='logpdf', x=x)

    def cdf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.get_prop(prop='cdf', x=x)

    def ppf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.get_prop(prop='ppf', x=x)

    def isf(
            self,
            x: typing.Union[float, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        return self.get_prop(prop='isf', x=x)
