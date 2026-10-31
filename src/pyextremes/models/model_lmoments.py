import logging
import typing

import pandas as pd
import scipy.stats

from pyextremes.models.distribution import Distribution
from pyextremes.models.model_scipy import ScipyModel

logger = logging.getLogger(__name__)


class Lmoments(ScipyModel):
    """
    L-Moments (Lmoments) model.

    Built around the lmom_fit method provided by the lmoments3 package,
    which must be installed (e.g. pip install "pyextremes[lmoments]").

    """

    def __init__(
        self,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        distribution_kwargs: typing.Optional[dict] = None,
    ) -> None:
        super().__init__(
            extremes=extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
        )
        self.distribution = Distribution(
            extremes=self.extremes,
            distribution=distribution,
            fit_method="Lmoments",
            **(distribution_kwargs or {}),
        )
        self.fit()

    @property
    def name(self) -> str:
        return "Lmoments"
