import functools
import typing

import scipy.stats
from pyextremes.tests.test_base import TestBase


class KolmogorovSmirnov(TestBase):
    def __init__(
        self,
        extremes,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        fit_parameters: typing.Dict[str, float],
        significance_level: float = 0.05,
    ) -> None:
        """One-sample Kolmogorov-Smirnov test.

        Tests whether `extremes` follow `distribution` with `fit_parameters`.

        Parameters
        ----------
        extremes : array-like
            Extreme values.
        distribution : str or scipy.stats.rv_continuous
            Distribution name compatible with scipy.stats
            or a subclass of scipy.stats.rv_continuous.
            See https://docs.scipy.org/doc/scipy/reference/stats.html
        fit_parameters : dict
            Dictionary with fit parameters passed to the `distribution` '.cdf' method.
            E.g. {"c": 0.3, "loc": 10, "scale": 2} for `distribution`
            with shape, location, and scale parameters.
        significance_level : float, optional
            Significance level in the range (0, 1).
            By default 0.05

        """
        super().__init__(
            extremes=extremes,
            distribution=distribution,
            fit_parameters=fit_parameters,
            significance_level=significance_level,
        )

        # Carry out KS test
        kstest = scipy.stats.kstest(
            rvs=self.extremes,
            cdf=functools.partial(
                self.distribution.cdf,
                **self.fit_parameters,
            ),
        )
        self.test_statistic = kstest.statistic
        self.pvalue = kstest.pvalue

        # Calculate critical value
        self.critical_value = scipy.stats.ksone.ppf(
            1 - self.significance_level / 2, len(self.extremes)
        )

    @property
    def name(self) -> str:
        return "Kolmogorov-Smirnov"

    @property
    def null_hypothesis(self) -> str:
        return f"data follows {self.distribution.name} distribution"

    @property
    def alternative_hypothesis(self) -> str:
        return f"data doesn't follow {self.distribution.name} distribution"

    @property
    def success(self) -> bool:
        if self.test_statistic > self.critical_value:
            return False
        else:
            return True

    def _repr_body(self) -> typing.List[str]:  # pragma: no cover
        if self.success:
            test_result = self.null_hypothesis
        else:
            test_result = self.alternative_hypothesis
        return [
            f"Test statistic: D = {self.test_statistic}",
            f"p-value: {self.pvalue}",
            "",
            f"Significance level: {self.significance_level}",
            f"Critical value: {self.critical_value}",
            f"Critical region: reject null-hypothesis if D > {self.critical_value}",
            "",
            f"Test result: {test_result}",
        ]
