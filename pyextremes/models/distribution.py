import logging
import typing

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger()


class Distribution:
    """
    A distribution class compatible with pyextremes models.
    It is a wrapper around scipy.stats.rv_continous distributions.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html

    Parameters
    ----------
    extremes : pandas.Series
        Time series of transformed extreme events.
    distribution : str or scipy.stats.rv_continuous
        scipy.stats distribution name or a subclass of scipy.stats.rv_continuous
        See https://docs.scipy.org/doc/scipy/reference/stats.html
    kwargs
        Special keyword arguments specific to a distribution which hold certain parameters fixed.
        E.g. fc=0 holds shape parameter 'c' at 0 essentially eliminating it as an independent parameter
        of the distribution, reducting its degree of freedom (number of free parameters) by one.
        Similarly, floc=0 hold the location parameter 'loc' at 0.
        See documentation of a specific scipy.stats distribution for documentation of these parameters.
    """

    def __init__(
            self,
            extremes: pd.Series,
            distribution: typing.Union[str, scipy.stats.rv_continuous],
            **kwargs
    ) -> None:
        self.extremes = extremes

        logger.info('getting scipy.stats distribution or ensuring distribution is a subclass of rv_continuous')
        if isinstance(distribution, scipy.stats.rv_continuous):
            self.distribution = distribution
        elif isinstance(distribution, str):
            self.distribution = getattr(scipy.stats, distribution)
        else:
            raise TypeError(f'invalid type in {type(distribution)} for the \'distribution\' argument')

        logger.info('setting invalid value to -np.inf')
        self.distribution.badvalue = -np.inf

        logger.info('getting distribution parameters')
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
                message = '; '.join(
                    [
                        f'\'{key}\' is not a valid keyword argument for \'{self.distribution.name}\' distribution',
                        f'valid keyword arguments: {", ".join(valid_kwargs)}'
                    ]
                )
                raise KeyError(message)

        logger.info('collecting free parameters')
        self.free_parameters = []
        for parameter in self.distribution_parameters:
            if f'f{parameter}' not in self.fixed_parameters:
                self.free_parameters.append(parameter)

        logger.info('fitting the distribution using scipy.stats MLE method')
        self.mle_parameters = {
            self.distribution_parameters[i]: value for i, value in enumerate(self.distribution.fit(self.extremes))
        }

    def fit(self, data: np.ndarray) -> tuple:
        return self.distribution.fit(data=data, **self.fixed_parameters)

    @property
    def name(self) -> str:
        return self.distribution.name

    @property
    def number_of_parameters(self) -> int:
        return len(self.free_parameters)

    def __repr__(self) -> str:
        free_parameters = ', '.join(self.free_parameters)

        fixed_parameters = ', '.join([f'{key}={value:,.3f}' for key, value in self.fixed_parameters.items()])
        if fixed_parameters == '':
            fixed_parameters = 'all parameters are free'

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
        Calculate log-probability of distribution for a given set of distribution parameters.
        Calculated as a sum of log-prior and log-likelihood.
        Log-prior is calculated from normal distribution with location as
        corresponding MLE value and scale as 100.
        Log-likelihood is calculated as sum of logarithms of PDF values for a distribution with parameters
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
        free_parameters = {self.free_parameters[i]: value for i, value in enumerate(theta)}

        logger.debug('calculating logprior')
        logprior = 0
        for key, value in free_parameters.items():
            logprior += scipy.stats.norm.logpdf(x=value, loc=self.mle_parameters[key], scale=100)

        logger.debug('calculating loglikelihood')
        loglikelihood = sum(
            self.distribution.logpdf(x=self.extremes.values, **free_parameters, **self.fixed_parameters)
        )

        return logprior + loglikelihood

    def get_initial_state(self, n_walkers: int) -> np.ndarray:
        """
        Get initial positions of emcee sampler walkers.
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
            Array with initial positions of emcee sampler walkers.
        """

        logger.info(f'getting initial positions for {n_walkers} walkers')
        mle_parameters = [self.mle_parameters[key] for key in self.free_parameters]
        return scipy.stats.norm.rvs(loc=mle_parameters, scale=0.01, size=(n_walkers, self.number_of_parameters))


if __name__ == '__main__':
    import pathlib
    import os
    from pyextremes import EVA
    test_path = pathlib.Path(os.getcwd()) / 'tests' / 'data' / 'battery_wl.csv'
    test_data = pd.read_csv(test_path, index_col=0, parse_dates=True, squeeze=True)
    test_data = (
        test_data
        .sort_index(ascending=True)
        .dropna()
    )
    test_data = test_data.loc[pd.to_datetime('1925'):]
    test_data = test_data - (test_data.index.array - pd.to_datetime('1992')) / pd.to_timedelta('1Y') * 2.87e-3
    eva = EVA(data=test_data)
    eva.get_extremes(method='BM', extremes_type='high', block_size='1Y', errors='ignore')

    self = Distribution(extremes=eva.extremes, distribution='genextreme')
