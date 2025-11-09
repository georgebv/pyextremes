import logging

from pyextremes.models.model_scipy import ScipyModel

logger = logging.getLogger(__name__)


class MLE(ScipyModel):
    """
    Maximum Likelihood Estimate (MLE) model.

    Built around the scipy.stats.rv_continuous.fit method.

    """

    @property
    def name(self) -> str:
        return "MLE"
