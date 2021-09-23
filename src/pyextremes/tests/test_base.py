import typing

from abc import ABC, abstractmethod

import numpy as np
import scipy.stats


class TestBase(ABC):
    def __init__(
        self,
        extremes,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        fit_parameters: typing.Dict[str, float],
        significance_level: float = 0.05,
    ) -> None:
        """Perform a statistical test.

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
        self.extremes = np.sort(np.asarray(extremes, dtype=np.float64).copy())

        if isinstance(distribution, scipy.stats.rv_continuous):
            self.distribution = distribution
        elif isinstance(distribution, str):
            self.distribution = getattr(scipy.stats, distribution)
        else:
            raise TypeError(
                f"invalid type in {type(distribution)} "
                f"for the 'distribution' argument, "
                f"must be string or scipy.stats.rv_continuous"
            )

        self.fit_parameters = fit_parameters
        self.significance_level = significance_level

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def null_hypothesis(self) -> str:
        pass

    @property
    @abstractmethod
    def alternative_hypothesis(self) -> str:
        pass

    @property
    @abstractmethod
    def success(self) -> bool:
        pass

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        summary = [
            f"{self.name} Test",
            "",
            f"Null hypothesis: {self.null_hypothesis}",
            f"Alternative hypothesis: {self.alternative_hypothesis}",
            "",
            *self._repr_body,
        ]

        longest_row = max(map(len, summary))
        summary[1] = "-" * longest_row
        summary.append(summary[1])
        summary[0] = " " * ((longest_row - len(summary[0])) // 2) + summary[0]
        for i, row in enumerate(summary):
            summary[i] += " " * (longest_row - len(row))

        return "\n".join(summary)

    @property
    @abstractmethod
    def _repr_body(self) -> typing.List[str]:
        pass
