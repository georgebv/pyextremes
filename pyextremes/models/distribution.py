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
import typing

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger()


class Distribution:
    """
    A distribution class compatible with pyextremes models.
    It is a wrapper around the scipy.stats.rv_continous class and its subclasses.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html

    Parameters
    ----------
    extremes : pandas.Series
        Time series of extreme events.
    distribution : str or scipy.stats.rv_continuous
        distribution name compatible with scipy.stats or a subclass of scipy.stats.rv_continuous
        See https://docs.scipy.org/doc/scipy/reference/stats.html for a list of continuous distributions
    kwargs
        Special keyword arguments, passsed to the .fit method of the continuous distribution.
        These keyword arguments represent parameters to be held fixed and must be shape, scale, or location
        parameter names with sufix 'f', e.g. 'fc', 'floc', or 'fscale'. By default, no parameters are fixed.
        See documentation of a specific scipy.stats distribution for names of available parameters.
        Examples:
            dict(fc=0) holds shape parameter 'c' at 0, essentially eliminating it as an independent parameter
                of the distribution, reducting its degree of freedom (number of free parameters) by one.
            dict(floc=0) hold the location parameter 'loc' at 0
            dict(fc=0, floc=10) holds shape and location parameters fixed at 0 and 10 respectively
    """

    def __init__(
            self,
            extremes: pd.Series,
            distribution: typing.Union[str, scipy.stats.rv_continuous],
            **kwargs
    ) -> None:
        self.extremes = extremes

        logger.info('fetching scipy.stats distribution or ensuring distribution is a subclass of rv_continuous')
        if isinstance(distribution, scipy.stats.rv_continuous):
            self.distribution = distribution
        elif isinstance(distribution, str):
            self.distribution = getattr(scipy.stats, distribution)
            if not isinstance(self.distribution, scipy.stats.rv_continuous):
                raise ValueError(f'\'{distribution}\' is not a continuous distribution')
        else:
            raise TypeError(f'invalid type in {type(distribution)} for the \'distribution\' argument')

        self.distribution.badvalue = -np.inf

        logger.info('getting a list of distribution parameter names')
        self.distribution_parameters = []
        if self.distribution.shapes is not None:
            self.distribution_parameters.extend([shape.strip() for shape in self.distribution.shapes.split(',')])
        self.distribution_parameters.extend(['loc', 'scale'])
        valid_kwargs = [f'f{parameter}' for parameter in self.distribution_parameters]

        logger.info('collecting fixed parameters')
        self.fixed_parameters = {}
        for key, value in kwargs.items():
            if key in valid_kwargs:
                self.fixed_parameters[key] = value
            else:
                raise TypeError(
                    f'\'{key}\' is not a valid keyword argument for \'{self.distribution.name}\' distribution, '
                    f'valid keyword arguments: {", ".join(valid_kwargs)}'
                )
        self._fixed_parameters = {key[1:]: value for key, value in self.fixed_parameters.items()}
        if len(self.fixed_parameters) == len(self.distribution_parameters):
            raise ValueError('all parameters of the distribution are fixed, there is nothing to fit')

        logger.info('collecting free parameters')
        self.free_parameters = []
        for parameter in self.distribution_parameters:
            if parameter not in self._fixed_parameters:
                self.free_parameters.append(parameter)

        logger.info('fitting the distribution using scipy.stats.rv_continuous.fit method')
        self.mle_parameters = self.fit(data=self.extremes.values)

    def fit(self, data: np.ndarray) -> dict:
        """
        Fit distribution to data using the scipy.stats.rv_continuous.fit method.
        Returns a dictionary with Maximum Likelihood Estimates of free distribution parameters.

        Parameters
        ----------
        data : numpy.ndarray
            Array with data to which the distribution is fit.

        Returns
        -------
        parameters : dict
            Dictionary with MLE of free distribution parameters.
        """

        logger.debug('calculating full MLE of distribution parameters')
        full_mle = self.distribution.fit(data=data, **self.fixed_parameters)

        logger.debug('packing distribution parameters into ordered free distribution parameters')
        free_parameters = {}
        for i, parameter in enumerate(self.distribution_parameters):
            if parameter in self.free_parameters:
                free_parameters[parameter] = full_mle[i]
            else:
                assert np.isclose(full_mle[i], self._fixed_parameters[parameter])
        return free_parameters

    @property
    def name(self) -> str:
        return self.distribution.name

    @property
    def number_of_parameters(self) -> int:
        return len(self.free_parameters)

    def __repr__(self) -> str:
        free_parameters = ', '.join(self.free_parameters)

        if len(self.fixed_parameters) == 0:
            fixed_parameters = 'all parameters are free'
        else:
            fixed_parameters = ', '.join([f'{key}={value:,.3f}' for key, value in self.fixed_parameters.items()])

        mle_parameters = ', '.join([f'{key}={value:,.3f}' for key, value in self.mle_parameters.items()])

        return '\n'.join(
            [
                'pyextremes distribution',
                '='*23,
                f'name: {self.name}',
                f'free parameters: {free_parameters}',
                f'fixed parameters: {fixed_parameters}',
                f'MLE parameters: {mle_parameters}'
            ]
        )

    def log_probability(self, theta: tuple) -> float:
        """
        Calculate log-probability of distribution for a given free distribution parameters' values
        as a sum of log-prior and log-likelihood.
        Log-prior is set to 0, which corresponds to an uninformative prior.
        Log-likelihood is calculated as sum of logpdf values for a distribution with parameters
        set to theta and values being extreme values.

        Parameters
        ----------
        theta : tuple
            Tuple with values of free distribution parameters.

        Returns
        -------
        logprobability : float
            log-probability for given theta.
        """

        logger.debug('unpacking theta')
        assert len(theta) == self.number_of_parameters, f'invalid theta size {len(theta):d}'
        free_parameters = dict(zip(self.free_parameters, theta))

        logger.debug('calculating log-likelihood')
        return sum(self.distribution.logpdf(x=self.extremes.values, **free_parameters, **self._fixed_parameters))

    def get_initial_state(self, n_walkers: int) -> np.ndarray:
        """
        Get initial positions of the ensemble sampler walkers.
        Positions are sampled from a normal distribution for each of the free distribution parameters
        (e.g. c, loc, scale) with location for each of them taken from scipy.stats MLE fit
        and standard deviation being 0.01.

        Parameters
        ----------
        n_walkers : int
            Number of walkers used by the emcee sampler.

        Returns
        -------
        initial_positions : numpy.ndarray
            Array with initial positions of the ensemble sampler walkers.
        """

        logger.info(f'getting initial positions for {n_walkers} walkers')
        mle_parameters = [self.mle_parameters[key] for key in self.free_parameters]
        return scipy.stats.norm.rvs(loc=mle_parameters, scale=0.01, size=(n_walkers, self.number_of_parameters))

    def free2full_parameters(
            self,
            free_parameters: typing.Union[dict, tuple, list, np.ndarray]
    ) -> np.ndarray:
        """
        Convert a dictionary or array-like object of free parameters to an array with full parameters.
        The output array has the same number of dimensions as input (1D for dict, tuple, list, or 1D numpu array).

        Parameters
        ----------
        free_parameters : dict or array-like
            Free parameters.
            If dict, then should be {parameter: value}. E.g. {'loc': 0, 'scale': 1}
            If array-like, should be either 1D or have shape of (n, number_of_free_parameters).
                E.g. [0, 1] for [loc, scale] if loc and scale are the only free parameters
                or [[0, 1], [2, 3],...] for a sequence of [loc, scale] pairs

        Returns
        -------
        full_parameters : numpy.ndarray
            Array with full parameters.
            1D for dict or 1D array
            (n, number_of_parameters) for array with shape (n, number_of_free_parameters)
        """

        if isinstance(free_parameters, dict):
            logger.debug('unpacking a free parameter dictionary')
            if len(free_parameters) != self.number_of_parameters:
                raise ValueError(f'invalid size in {len(free_parameters)} for the \'free_parameters\' argument')
            full_parameters = []
            for parameter in self.distribution_parameters:
                try:
                    full_parameters.append(free_parameters[parameter])
                except KeyError:
                    full_parameters.append(self._fixed_parameters[parameter])
            return np.array(full_parameters)

        elif isinstance(free_parameters, (tuple, list, np.ndarray)):
            logger.debug('unpacking a free parameter array')
            free_parameters = np.array(free_parameters)

            if len(free_parameters.shape) == 1:
                logger.debug('unpacking 1D free parameter array')
                if len(free_parameters) != self.number_of_parameters:
                    raise ValueError(
                        f'invalid number of free parameters in {len(free_parameters):d} '
                        f'for the \'free_parameters\' array, must be {self.number_of_parameters:d}'
                    )
                i = 0
                full_parameters = []
                for parameter in self.distribution_parameters:
                    try:
                        full_parameters.append(self._fixed_parameters[parameter])
                    except KeyError:
                        full_parameters.append(free_parameters[i])
                        i += 1
                return np.array(full_parameters)

            elif len(free_parameters.shape) == 2:
                logger.debug('unpacking 2D free parameter')
                if free_parameters.shape[1] != self.number_of_parameters:
                    raise ValueError(
                        f'invalid number of free parameters in {free_parameters.shape[1]:d} '
                        f'for the \'free_parameters\' array, must be {self.number_of_parameters:d}'
                    )
                j = 0
                full_parameters = np.zeros(shape=(len(free_parameters), len(self.distribution_parameters)))
                for i, parameter in enumerate(self.distribution_parameters):
                    try:
                        full_parameters[:, i] = self._fixed_parameters[parameter]
                    except KeyError:
                        full_parameters[:, i] = free_parameters[:, j]
                        j += 1
                return full_parameters

            else:
                raise ValueError(f'invalid shape in {free_parameters.shape} for the \'free_parameters\' array')

        else:
            raise TypeError(f'invalid type in {type(free_parameters)} for the \'free_parameters\' argument')

    def get_prop(
            self,
            prop: str,
            x: typing.Union[int, float, tuple, list, np.ndarray],
            free_parameters: typing.Union[dict, tuple, list, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        """
        Calculate a property such as isf, cdf, or logpdf.

        Parameters
        ----------
        prop : str
            Property name (e.g. 'isf' or 'logpdf').
        x : float or array-like
            Data for which the property is calculated.
            Scalar or 1D array-like.
        free_parameters : dict or numpy.ndarray
            Dictionary or array with free distribution parameter values.
            See self.free2full_parameters method documentation.

        Returns
        -------
        result : float or numpy.ndarray
            Output of property.
            If x is scalar:
                output is scalar or 1D array with length equal to number of free_parameter combinations
                    for 1D free_parameters=[1, 2] it is scalar
                    for 2D free parameters=[[1, 2], [3, 4], [5, 6]] it is an array of length len(free_paramters)
            If x is a 1D array:
                output is a 1D or 2D array
                    for free_parameters=[1, 2] it is a 1D array of length len(x)
                    for free_parameters=[[1, 2], [3, 4], ...] it is a 2D array of shape (len(x), len(free_parameters)
        """

        logger.debug('getting property function')
        prop_function = getattr(self.distribution, prop)

        logger.debug('converting free_parameters to an array of full parameters')
        full_parameters = self.free2full_parameters(free_parameters=free_parameters)

        if isinstance(x, (int, float)):
            logger.debug('calculating property for scalar x')
            return prop_function(x, *np.transpose(full_parameters))
        elif isinstance(x, (tuple, list, np.ndarray)):
            logger.debug('calculating property for array x')
            x = np.array(x)
            if len(x.shape) != 1:
                raise ValueError(f'invalid shape in {x.shape} for the \'x\' argument')

            if len(full_parameters.shape) == 1:
                logger.debug('calculating property for one combination of parameters')
                return prop_function(x, *full_parameters)

            elif len(full_parameters.shape) == 2:
                logger.debug(f'calculating property for {len(full_parameters)} combinations of parameters')
                full_x = np.tile(x, reps=(len(full_parameters), 1))
                return prop_function(np.transpose(full_x), *np.transpose(full_parameters))

            else:
                raise RuntimeError('this is a bug: self.free2full_parameters method returned invalid value')
        else:
            raise TypeError(f'invalid type in {type(x)} for the \'x\' argument')
