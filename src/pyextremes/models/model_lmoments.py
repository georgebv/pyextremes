import logging
import typing

import pandas as pd
import scipy.stats

from pyextremes.models.model_scipy import ScipyModel

logger = logging.getLogger(__name__)

class Lmoments(ScipyModel):
    def __init__(
        self,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        distribution_kwargs: typing.Optional[dict] = None,
    ) -> None:
        """
        L-Moments (Lmoments) model.

        Built around the scipy.stats.rv_continuous.fit method.

        """
        super().__init__(
            extremes=extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            method="Lmoments"
        )

    @property
    def name(self) -> str:
        return "Lmoments"