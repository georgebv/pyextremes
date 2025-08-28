import itertools
import logging
import multiprocessing
import os
import typing

import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.models.model_base import AbstractModelBaseClass
from pyextremes.models.model_scipy import ScipyModel

logger = logging.getLogger(__name__)

class LMOM(ScipyModel):
    def __init__(
        self,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        distribution_kwargs: typing.Optional[dict] = None,
    ) -> None:
        """
        L-Moments (LMOM) model.

        Built around the scipy.stats.rv_continuous.fit method.

        """
        super().__init__(
            extremes=extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            method="LMOM"
        )

    @property
    def name(self) -> str:
        return "LMOM"