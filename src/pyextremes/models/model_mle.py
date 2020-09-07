import logging
import typing

import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.models.model_base import AbstractModelBaseClass

logger = logging.getLogger(__name__)


class MLE(AbstractModelBaseClass):
    def __init__(
        self,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        distribution_kwargs: typing.Optional[dict] = None,
    ) -> None:
        """
        Maximum Likelihood Estimate (MLE) model.

        Built around the scipy.stats.rv_continuous.fit method.

        """
        super().__init__(
            extremes=extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
        )

        # Initialize 'fit_parameter_cache'
        self.fit_parameter_cache: typing.List[tuple] = []

    @property
    def name(self) -> str:
        return "MLE"

    def fit(self, **kwargs) -> None:
        if len(kwargs) != 0:
            raise TypeError(
                f"unrecognized arguments passed in: {', '.join(kwargs.keys())}"
            )
        self.fit_parameters = self.distribution.mle_parameters
        logger.debug(
            f"fit {self.distribution.name} distribution "
            f"with parameters {self.distribution.mle_parameters}"
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
            Width of confidence interval (0, 1) (default=None).
            If None (default), return None
            for upper and lower confidence interval bounds.
        kwargs
            n_samples : int, optional
                Number of bootstrap samples used to estimate
                confidence interval bounds (default=100).

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
        n_samples = kwargs.pop("n_samples", 100)
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
                n_samples,
            )
            try:
                # Try to fetch pre-calculated values from cache
                rv, cil, ciu = self.return_value_cache[key]
                logger.debug(
                    f"fetched return value for {key} from cache as {(rv, cil, ciu)}"
                )
            except KeyError:
                # Value not in cache - calculate new return value
                rv = self.distribution.distribution.isf(
                    q=ep, **self.fit_parameters, **self.distribution._fixed_parameters
                )

                # Calculate confidence intervals
                if alpha is None:
                    cil = None
                    ciu = None
                else:
                    # If cache doesn't have enough values, calculate new fit parameters
                    if len(self.fit_parameter_cache) < n_samples:
                        for _ in range(n_samples - len(self.fit_parameter_cache)):
                            sample = np.random.choice(
                                a=self.extremes.values,
                                size=len(self.extremes),
                                replace=True,
                            )
                            self.fit_parameter_cache.append(
                                self.distribution.distribution.fit(
                                    data=sample, **self.distribution.fixed_parameters
                                )
                            )
                    # Calculate confidence intervals
                    rv_sample = self.distribution.distribution.isf(
                        ep, *np.transpose(self.fit_parameter_cache)
                    )
                    cil, ciu = np.quantile(
                        a=rv_sample, q=[(1 - alpha) / 2, (1 + alpha) / 2]
                    )

                # Add calculated return value and intervals to cache
                self.return_value_cache[key] = (rv, cil, ciu)
                logger.debug(f"calculated return value for {key} as {(rv, cil, ciu)}")

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
            "MLE model",
            "",
            f"free parameters: {free_parameters}",
            f"fixed parameters: {fixed_parameters}",
            f"AIC: {self.AIC:.3f}",
            f"loglikelihood: {self.loglikelihood:.3f}",
            f"return value cache size: {len(self.return_value_cache):,d}",
            f"fit parameter cache size: {len(self.fit_parameter_cache):,d}",
        ]

        longest_row = max(map(len, summary))
        summary[1] = "-" * longest_row
        summary.append(summary[1])
        summary[0] = " " * ((longest_row - len(summary[0])) // 2) + summary[0]
        for i, row in enumerate(summary):
            summary[i] += " " * (longest_row - len(row))

        return "\n".join(summary)
