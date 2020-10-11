import os

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from pyextremes.models import MLE, get_model


@pytest.fixture(scope="function")
def extremes() -> pd.Series:
    np.random.seed(0)
    return pd.Series(
        index=pd.date_range(start="2000-01-01", periods=1000, freq="1H"),
        data=scipy.stats.genextreme.rvs(c=0, loc=10, scale=2, size=1000),
    )


@pytest.fixture(scope="function")
def mle_model(extremes) -> MLE:
    return get_model(
        model="MLE",
        extremes=extremes,
        distribution="genextreme",
        distribution_kwargs={"fc": 0},
    )


class TestMLE:
    def test_model(self, extremes, mle_model):
        # Test extremes attribute
        assert np.all(mle_model.extremes.index == extremes.index)
        assert np.allclose(mle_model.extremes.values, extremes.values)

        # Test fit_parameters attribute
        assert mle_model.distribution.name == "genextreme"
        assert len(mle_model.fit_parameters) == 2
        for key, value in {"loc": 10, "scale": 2}.items():
            assert key in mle_model.fit_parameters
            assert np.isclose(mle_model.fit_parameters[key], value, rtol=0, atol=0.1)

        # Test trace attribute
        with pytest.raises(TypeError, match=r"trace property is not"):
            mle_model.trace

        # Test return_value_cache attribute
        assert isinstance(mle_model.return_value_cache, dict)
        assert len(mle_model.return_value_cache) == 0

        # Test name attribute
        assert mle_model.name == "MLE"

        # Test fit_parameter_cache attribute
        assert isinstance(mle_model.fit_parameter_cache, list)
        assert len(mle_model.fit_parameter_cache) == 0

    def test_fit(self, mle_model):
        with pytest.raises(TypeError, match=r"unrecognized arguments.*bad_argument"):
            mle_model.fit(bad_argument=1)

    def test_loglikelihood(self, extremes, mle_model):
        assert np.allclose(
            np.sum(
                scipy.stats.genextreme.logpdf(
                    x=extremes.values,
                    **mle_model.fit_parameters,
                    **mle_model.distribution._fixed_parameters,
                )
            ),
            mle_model.loglikelihood,
            rtol=0,
            atol=0.01,
        )

    def test_aic(self, extremes, mle_model):
        k = 2
        n = len(extremes)
        aic = 2 * k - 2 * mle_model.loglikelihood
        correction = (2 * k ** 2 + 2 * k) / (n - k - 1)
        assert np.isclose(mle_model.AIC, aic + correction, rtol=0, atol=0.01)

    @pytest.mark.parametrize("prop", ["pdf", "logpdf", "cdf", "ppf", "isf"])
    def test_properties(self, prop, mle_model):
        assert np.isclose(
            getattr(mle_model, prop)(x=0.1),
            getattr(scipy.stats.genextreme, prop)(0.1, c=0, loc=10, scale=2),
            rtol=0.1,
            atol=0.1,
        )

    def test_get_return_value(self, mle_model):
        with pytest.raises(TypeError, match=r"unrecognized arguments.*bad_argument"):
            mle_model.get_return_value(exceedance_probability=0.1, bad_argument=1)
        with pytest.raises(ValueError, match=r"invalid shape.*exceedance_probability"):
            mle_model.get_return_value(exceedance_probability=[[1, 2, 3], [1, 2, 3]])
        with pytest.raises(
            ValueError, match=r"invalid value.*n_samples.*must be positive"
        ):
            mle_model.get_return_value(exceedance_probability=0.1, n_samples=-1)

        # Test scalar, no alpha
        rv, cil, ciu = mle_model.get_return_value(exceedance_probability=0.1)
        assert np.isclose(
            rv, scipy.stats.genextreme.isf(0.1, c=0, loc=10, scale=2), rtol=0, atol=0.1
        )
        assert np.isnan(cil) and np.isnan(ciu)
        assert len(mle_model.return_value_cache) == 1
        assert len(mle_model.fit_parameter_cache) == 0
        assert len(np.unique(mle_model.fit_parameter_cache, axis=0)) == len(
            mle_model.fit_parameter_cache
        )
        seed_cahe_size = 0
        assert len(mle_model.seed_cache) == seed_cahe_size

        # Test scalar, with alpha
        rv, cil, ciu = mle_model.get_return_value(
            exceedance_probability=0.1, alpha=0.95
        )
        assert np.isclose(
            rv,
            scipy.stats.genextreme.isf(0.1, c=0, loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not (np.isnan(cil) or np.isnan(ciu))
        assert len(mle_model.return_value_cache) == 2
        assert len(mle_model.fit_parameter_cache) == 100
        assert len(np.unique(mle_model.fit_parameter_cache, axis=0)) == len(
            mle_model.fit_parameter_cache
        )
        seed_cahe_size = min(2, os.cpu_count())
        assert len(mle_model.seed_cache) == seed_cahe_size

        # Test array, with alpha
        rv, cil, ciu = mle_model.get_return_value(
            exceedance_probability=[0.1, 0.2], alpha=0.95
        )
        assert np.allclose(
            rv,
            scipy.stats.genextreme.isf([0.1, 0.2], c=0, loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not np.any(np.isnan(cil) | np.isnan(ciu))
        assert len(mle_model.return_value_cache) == 3
        assert len(mle_model.fit_parameter_cache) == 100
        assert len(np.unique(mle_model.fit_parameter_cache, axis=0)) == len(
            mle_model.fit_parameter_cache
        )
        seed_cahe_size = min(2, os.cpu_count())
        assert len(mle_model.seed_cache) == seed_cahe_size

        # Test small additional n_samples
        rv, cil, ciu = mle_model.get_return_value(
            exceedance_probability=0.1,
            alpha=0.95,
            n_samples=120,
        )
        assert np.isclose(
            rv,
            scipy.stats.genextreme.isf(0.1, c=0, loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not (np.isnan(cil) or np.isnan(ciu))
        assert len(mle_model.return_value_cache) == 4
        assert len(np.unique(mle_model.fit_parameter_cache, axis=0)) == len(
            mle_model.fit_parameter_cache
        )
        seed_cahe_size = min(2, os.cpu_count()) + 1
        assert len(mle_model.seed_cache) == seed_cahe_size

        # Test large additional n_samples, not multiple of 50
        rv, cil, ciu = mle_model.get_return_value(
            exceedance_probability=0.1,
            alpha=0.95,
            n_samples=201,
        )
        assert np.isclose(
            rv,
            scipy.stats.genextreme.isf(0.1, c=0, loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not (np.isnan(cil) or np.isnan(ciu))
        assert len(mle_model.return_value_cache) == 5
        assert len(mle_model.fit_parameter_cache) == 201
        assert len(np.unique(mle_model.fit_parameter_cache, axis=0)) == len(
            mle_model.fit_parameter_cache
        )
        seed_cahe_size += min(2, os.cpu_count())
        assert len(mle_model.seed_cache) == seed_cahe_size

    def test_repr(self, mle_model):
        repr_value = str(mle_model)
        assert len(repr_value.split("\n")) == 9

    @pytest.mark.parametrize(
        "distribution_name, distribution_kwargs, scipy_parameters",
        [
            ("genextreme", {}, (0.5, 10, 2)),
            ("gumbel_r", {}, (10, 2)),
            ("genpareto", {}, (0.5, 0, 2)),
            ("genpareto", {"floc": 0}, (0.5, 0, 2)),
            ("expon", {}, (0, 2)),
            ("expon", {"floc": 0}, (0, 2)),
        ],
    )
    def test_mle_distributions(
        self, distribution_name, distribution_kwargs, scipy_parameters
    ):
        scipy_distribution = getattr(scipy.stats, distribution_name)

        np.random.seed(0)
        extremes = pd.Series(
            index=pd.date_range(start="2000-01-01", periods=100, freq="1H"),
            data=scipy_distribution.rvs(*scipy_parameters, size=100),
        )
        model = get_model(
            model="MLE",
            extremes=extremes,
            distribution=distribution_name,
            distribution_kwargs=distribution_kwargs,
        )

        # Test extremes attribute
        assert np.all(model.extremes.index == extremes.index)
        assert np.allclose(model.extremes.values, extremes.values)

        # Test fit_parameters attribute
        assert model.distribution.name == distribution_name
        assert len(model.fit_parameters) == len(scipy_parameters) - len(
            distribution_kwargs
        )

        # Test trace attribute
        with pytest.raises(TypeError, match=r"trace property is not"):
            model.trace

        # Test return_value_cache attribute
        assert isinstance(model.return_value_cache, dict)
        assert len(model.return_value_cache) == 0

        # Test name attribute
        assert model.name == "MLE"

        # Test fit_parameter_cache attribute
        assert isinstance(model.fit_parameter_cache, list)
        assert len(model.fit_parameter_cache) == 0

        # Test loglikelihood
        assert np.isclose(
            model.loglikelihood,
            np.sum(
                scipy_distribution.logpdf(
                    model.extremes.values,
                    **model.fit_parameters,
                    **model.distribution._fixed_parameters,
                )
            ),
            rtol=0,
            atol=0.01,
        )

        # Test AIC
        k = model.distribution.number_of_parameters
        n = len(model.extremes)
        loglikelihood = sum(
            scipy_distribution.logpdf(
                model.extremes.values,
                **model.fit_parameters,
                **model.distribution._fixed_parameters,
            )
        )
        aic = 2 * k - 2 * loglikelihood
        correction = (2 * k ** 2 + 2 * k) / (n - k - 1)
        assert np.isclose(model.AIC, aic + correction)

        # Test properties
        for prop in ["pdf", "cdf", "ppf", "isf", "logpdf"]:
            assert np.isclose(
                getattr(model, prop)(0.1),
                getattr(scipy_distribution, prop)(
                    0.1, **model.fit_parameters, **model.distribution._fixed_parameters
                ),
                rtol=0.1,
                atol=0.1,
            )

        # Test repr
        repr_value = str(model)
        assert len(repr_value.split("\n")) == 9
        if len(distribution_kwargs) == 0:
            assert "all parameters are free" in repr_value

        # Test get_return_value
        with pytest.raises(TypeError, match=r"unrecognized arguments.*bad_argument"):
            model.get_return_value(exceedance_probability=0.1, bad_argument=1)
        with pytest.raises(ValueError, match=r"invalid shape.*exceedance_probability"):
            model.get_return_value(exceedance_probability=[[1, 2, 3], [1, 2, 3]])
        with pytest.raises(
            ValueError, match=r"invalid value.*n_samples.*must be positive"
        ):
            model.get_return_value(exceedance_probability=0.1, n_samples=-1)

        # Test scalar, no alpha
        rv, cil, ciu = model.get_return_value(exceedance_probability=0.1)
        assert np.isclose(
            rv,
            scipy_distribution.isf(
                0.1, **model.fit_parameters, **model.distribution._fixed_parameters
            ),
            rtol=0,
            atol=0.1,
        )
        assert np.isnan(cil) and np.isnan(ciu)
        assert len(model.return_value_cache) == 1
        assert len(model.fit_parameter_cache) == 0
        assert len(np.unique(model.fit_parameter_cache, axis=0)) == len(
            model.fit_parameter_cache
        )
        seed_cahe_size = 0
        assert len(model.seed_cache) == seed_cahe_size

        # Test scalar, with alpha
        rv, cil, ciu = model.get_return_value(exceedance_probability=0.1, alpha=0.95)
        assert np.isclose(
            rv,
            scipy_distribution.isf(
                0.1, **model.fit_parameters, **model.distribution._fixed_parameters
            ),
            rtol=0,
            atol=0.1,
        )
        assert not (np.isnan(cil) or np.isnan(ciu))
        assert len(model.return_value_cache) == 2
        assert len(model.fit_parameter_cache) == 100
        assert len(np.unique(model.fit_parameter_cache, axis=0)) == len(
            model.fit_parameter_cache
        )
        seed_cahe_size = min(2, os.cpu_count())
        assert len(model.seed_cache) == seed_cahe_size

        # Test array, with alpha
        rv, cil, ciu = model.get_return_value(
            exceedance_probability=[0.1, 0.2], alpha=0.95
        )
        assert np.allclose(
            rv,
            scipy_distribution.isf(
                [0.1, 0.2],
                **model.fit_parameters,
                **model.distribution._fixed_parameters,
            ),
            rtol=0,
            atol=0.1,
        )
        assert not np.any(np.isnan(cil) | np.isnan(ciu))
        assert len(model.return_value_cache) == 3
        assert len(model.fit_parameter_cache) == 100
        assert len(np.unique(model.fit_parameter_cache, axis=0)) == len(
            model.fit_parameter_cache
        )
        seed_cahe_size = min(2, os.cpu_count())
        assert len(model.seed_cache) == seed_cahe_size
