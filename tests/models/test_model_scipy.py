import lmoments3.distr
import numpy as np
import pandas as pd
import pytest
import scipy.stats

from pyextremes.models import get_model
from pyextremes.models.model_scipy import get_fit_parameters


@pytest.fixture(scope="function")
def extremes() -> pd.Series:
    np.random.seed(0)
    return pd.Series(
        index=pd.date_range(start="2000-01-01", periods=100, freq="1h"),
        data=scipy.stats.gumbel_r.rvs(loc=10, scale=2, size=100),
    )


def _reference_fit(model: str, data: np.ndarray) -> tuple:
    if model == "MLE":
        return scipy.stats.gumbel_r.fit(data, method="mle")
    if model == "MOM":
        return scipy.stats.gumbel_r.fit(data, method="mm")
    if model == "Lmoments":
        return tuple(lmoments3.distr.gum.lmom_fit(data=data).values())
    raise AssertionError


@pytest.mark.parametrize("model", ["MLE", "MOM", "Lmoments"])
def test_bootstrap_uses_model_fit_method(extremes, model):
    fitted_model = get_model(model=model, extremes=extremes, distribution="gumbel_r")
    seed = 12345
    n = 5
    fit_parameters = get_fit_parameters(
        params=(
            n,
            fitted_model.distribution.distribution,
            extremes.values,
            fitted_model.distribution.fit_method,
            fitted_model.distribution.fixed_parameters,
            seed,
        )
    )
    assert len(fit_parameters) == n

    # Reproduce bootstrap samples and fit them using the expected method
    rng_generator = np.random.default_rng(seed=seed)
    for parameters in fit_parameters:
        sample = rng_generator.choice(a=extremes.values, size=len(extremes))
        assert np.allclose(parameters, _reference_fit(model, sample))


@pytest.mark.parametrize("model", ["MOM", "Lmoments"])
def test_bootstrap_confidence_intervals_differ_from_mle(extremes, model):
    intervals = {}
    for name in ["MLE", model]:
        np.random.seed(0)
        fitted_model = get_model(model=name, extremes=extremes, distribution="gumbel_r")
        _, ci_lower, ci_upper = fitted_model.get_return_value(
            exceedance_probability=0.01,
            alpha=0.95,
            n_samples=20,
        )
        intervals[name] = (ci_lower, ci_upper)
    assert not np.allclose(intervals["MLE"], intervals[model])
