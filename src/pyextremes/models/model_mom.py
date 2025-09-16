import logging
import typing

import pandas as pd
import scipy.stats

from pyextremes.models.model_scipy import ScipyModel

logger = logging.getLogger(__name__)

class MOM(ScipyModel):
    def __init__(
        self,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        distribution_kwargs: typing.Optional[dict] = None,
    ) -> None:
        """
        Method of Moments (MOM) model.

        Built around the scipy.stats.rv_continuous.fit method.

        """

        super().__init__(
            extremes=extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            method="MOM"
        )

    @property
    def name(self) -> str:
        return "MOM"