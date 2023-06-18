import logging
import typing
import warnings

import emcee
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)


class Emcee(AbstractModelBaseClass):
    def __init__(
        self,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        distribution_kwargs: typing.Optional[dict] = None,
        n_walkers: int = 100,
        n_samples: int = 500,
        progress: bool = False,
    ) -> None:
        """
        Markov Chain Monte Carlo (MCMC) model.

        Built around the 'emcee' package by Daniel Foreman-Mackey

        """
        super().__init__(
            extremes=extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            n_walkers=n_walkers,
            n_samples=n_samples,
            progress=progress,
        )
        self.n_walkers = n_walkers
        self.n_samples = n_samples

    @property
    def name(self) -> str:
        return "Emcee"

    def fit(self, **kwargs) -> None:
        # Parse kwargs
        n_walkers: int = kwargs.pop("n_walkers")
        n_samples: int = kwargs.pop("n_samples")
        progress: bool = kwargs.pop("progress")
        if len(kwargs) != 0:
            raise TypeError(
                f"unrecognized arguments passed in: {', '.join(kwargs.keys())}"
            )

        # Declare Emcee ensemble sampler
        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.distribution.number_of_parameters,
            log_prob_fn=self.distribution.log_probability,
        )

        # Run the ensemble sampler
        logger.debug(
            "running ensemble sampler with %d walkers and %d samples",
            n_walkers,
            n_samples,
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            sampler.run_mcmc(
                initial_state=self.distribution.get_initial_state(n_walkers=n_walkers),
                nsteps=n_samples,
                progress=progress,
            )
        logger.debug(
            "finished run for ensemble sampler with %d walkers and %d samples",
            n_walkers,
            n_samples,
        )

        # Extract ensemble sampler chain
        self._trace: np.ndarray = sampler.get_chain().transpose((1, 0, 2))

        # Calculate fit parameters as MAP of distribution parameters
        kernel = scipy.stats.gaussian_kde(np.vstack(self._trace).transpose())

        def kde_func(x):
            return -kernel(x)[0]

        fit_parameters = self._trace.mean(axis=(0, 1))
        solution = scipy.optimize.minimize(
            kde_func,
            x0=fit_parameters,
            method="Nelder-Mead",
        )
        if solution.success:
            fit_parameters = solution.x
        else:  # pragma: no cover
            warnings.warn(
                message=(
                    "cannot calculate MAP using Gaussian KDE, "
                    "setting fit parameters as mean"
                ),
                category=RuntimeWarning,
            )
        self._fit_parameters = dict(
            zip(self.distribution.free_parameters, fit_parameters)
        )

    @property
    def trace_map(self) -> tuple:
        return tuple(
            self.fit_parameters[parameter]
            for parameter in self.distribution.free_parameters
        )

    def get_return_value(
        self, exceedance_probability, alpha: typing.Optional[float] = None, **kwargs
    ) -> tuple:
        """
        Calculate return value and confidence interval bounds.

        Parameters
        ----------
        exceedance_probability : array-like
            Exceedance probability or 1D array of exceedance probabilities.
            Each exceedance probability must be in the [0, 1) range.
        alpha : float, optional
            Width of confidence interval (0, 1).
            If None (default), return None
            for upper and lower confidence interval bounds.
        kwargs
            burn_in : int, optional
                Burn-in value (number of first steps to discard for each walker).
                By default it is 0 (no values are discarded).

        Returns
        -------
        return_value : array-like
            Return values.
        ci_lower : array-like
            Lower confidence interval bounds.
        ci_upper : array-like
            Upper confidence interval bounds.

        """
        # Parse 'kwargs'
        burn_in = kwargs.pop("burn_in", 0)
        if len(kwargs) != 0:
            raise TypeError(
                f"unrecognized arguments passed in: {', '.join(kwargs.keys())}"
            )

        # Convert 'exceedance_probability' to ndarray
        exceedance_probability = np.asarray(
            a=exceedance_probability, dtype=np.float64
        ).copy()
        if exceedance_probability.ndim == 0:
            exceedance_probability = exceedance_probability[np.newaxis]
        if exceedance_probability.ndim != 1:
            raise ValueError(
                f"invalid shape in {exceedance_probability.shape} "
                f"for the 'exceedance_probability' argument, must be 1D array"
            )

        # Calculate return values
        return_value = np.full(
            shape=exceedance_probability.shape, fill_value=np.nan, dtype=np.float64
        )
        ci_lower = return_value.copy()
        ci_upper = return_value.copy()
        for i, ep in enumerate(exceedance_probability):
            key: typing.Tuple[float, typing.Optional[float], int] = (
                ep,
                alpha,
                burn_in,
            )
            try:
                # Try to fetch pre-calculated values from cache
                rv, cil, ciu = self.return_value_cache[key]
                logger.debug(
                    "fetched return value for %s from cache as (%s, %s, %s)",
                    key,
                    rv,
                    cil,
                    ciu,
                )
            except KeyError:
                # Value not in cache - calculate new return value
                rv = self.distribution.distribution.isf(
                    q=ep,
                    **self.fit_parameters,
                    **self.distribution._fixed_parameters,
                )

                # Calculate confidence intervals
                if alpha is None:
                    cil = None
                    ciu = None
                else:
                    # Calculate confidence intervals
                    rv_sample = self.distribution.get_prop(
                        prop="isf",
                        x=ep,
                        free_parameters=np.vstack(self.trace[:, burn_in:, :]),
                    )
                    cil, ciu = np.quantile(
                        a=rv_sample, q=[(1 - alpha) / 2, (1 + alpha) / 2]
                    )

                # Add calculated return value and intervals to cache
                self.return_value_cache[key] = (rv, cil, ciu)
                logger.debug(
                    "calculated return value for %s as (%s, %s, %s)",
                    key,
                    rv,
                    cil,
                    ciu,
                )

            return_value[i] = rv
            ci_lower[i] = cil
            ci_upper[i] = ciu

        # Return results
        if len(return_value) == 1:
            return return_value[0], ci_lower[0], ci_upper[0]
        else:
            return return_value, ci_lower, ci_upper

    def __repr__(self) -> str:
        free_parameters = ", ".join(
            [
                f"{parameter}={self.fit_parameters[parameter]:.3f}"
                for parameter in self.distribution.free_parameters
            ]
        )

        fixed_parameters = ", ".join(
            [
                f"{key}={value:.3f}"
                for key, value in self.distribution.fixed_parameters.items()
            ]
        )
        if fixed_parameters == "":
            fixed_parameters = "all parameters are free"

        summary = [
            "Emcee model",
            "",
            f"free parameters: {free_parameters}",
            f"fixed parameters: {fixed_parameters}",
            f"AIC: {self.AIC:.3f}",
            f"loglikelihood: {self.loglikelihood:.3f}",
            f"number of walkers: {self.n_walkers:d}",
            f"number of samples: {self.n_samples:d}",
            f"return value cache size: {len(self.return_value_cache):,d}",
        ]

        longest_row = max(map(len, summary))
        summary[1] = "-" * longest_row
        summary.append(summary[1])
        summary[0] = " " * ((longest_row - len(summary[0])) // 2) + summary[0]
        for i, row in enumerate(summary):
            summary[i] += " " * (longest_row - len(row))

        return "\n".join(summary)
