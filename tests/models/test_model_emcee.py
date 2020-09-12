import numpy as np
import pandas as pd
import pytest
import scipy.stats

from pyextremes.models import Emcee, get_model


@pytest.fixture(scope="function")
def extremes() -> pd.Series:
    np.random.seed(0)
    return pd.Series(
        index=pd.date_range(start="2000-01-01", periods=1000, freq="1H"),
        data=scipy.stats.genextreme.rvs(c=0, loc=10, scale=2, size=1000),
    )


@pytest.fixture(scope="function")
def emcee_model(extremes) -> Emcee:
    return get_model(
        model="Emcee",
        extremes=extremes,
        distribution="genextreme",
        distribution_kwargs={"fc": 0},
        n_walkers=20,
        n_samples=100,
    )


class TestEmcee:
    def test_model(self, extremes, emcee_model):
        # Test extremes attribute
        assert np.all(emcee_model.extremes.index == extremes.index)
        assert np.allclose(emcee_model.extremes.values, extremes.values)

        # Test fit_parameters attribute
        assert emcee_model.distribution.name == "genextreme"
        assert len(emcee_model.fit_parameters) == 2
        for key, value in {"loc": 10, "scale": 2}.items():
            assert key in emcee_model.fit_parameters
            assert np.isclose(emcee_model.fit_parameters[key], value, rtol=0, atol=0.1)

        # Test emcee attributes
        assert emcee_model.n_walkers == 20
        assert emcee_model.n_samples == 100

        # Test trace attribute
        assert emcee_model.trace.shape == (20, 100, 2)
        assert np.issubdtype(emcee_model.trace.dtype, np.float64)

        # Test return_value_cache attribute
        assert isinstance(emcee_model.return_value_cache, dict)
        assert len(emcee_model.return_value_cache) == 0

        # Test name attribute
        assert emcee_model.name == "Emcee"

    def test_fit(self, emcee_model):
        with pytest.raises(TypeError, match=r"unrecognized arguments.*bad_argument"):
            emcee_model.fit(
                n_walkers=20,
                n_samples=100,
                progress=False,
                bad_argument=1,
            )

    def test_trace_map(self, emcee_model):
        assert (
            len(emcee_model.trace_map) == emcee_model.distribution.number_of_parameters
        )
        trace_map = tuple(
            emcee_model.fit_parameters[parameter]
            for parameter in emcee_model.distribution.free_parameters
        )
        assert emcee_model.trace_map == trace_map

    def test_loglikelihood(self, extremes, emcee_model):
        assert np.allclose(
            np.sum(
                scipy.stats.genextreme.logpdf(
                    x=extremes.values,
                    **emcee_model.fit_parameters,
                    **emcee_model.distribution._fixed_parameters,
                )
            ),
            emcee_model.loglikelihood,
            rtol=0,
            atol=0.01,
        )

    def test_aic(self, extremes, emcee_model):
        k = 2
        n = len(extremes)
        aic = 2 * k - 2 * emcee_model.loglikelihood
        correction = (2 * k ** 2 + 2 * k) / (n - k - 1)
        assert np.isclose(emcee_model.AIC, aic + correction, rtol=0, atol=0.01)

    @pytest.mark.parametrize("prop", ["pdf", "logpdf", "cdf", "ppf", "isf"])
    def test_properties(self, prop, emcee_model):
        assert np.isclose(
            getattr(emcee_model, prop)(x=0.1),
            getattr(scipy.stats.genextreme, prop)(0.1, c=0, loc=10, scale=2),
            rtol=0.1,
            atol=0.1,
        )

    def test_get_return_value(self, emcee_model):
        with pytest.raises(TypeError, match=r"unrecognized arguments.*bad_argument"):
            emcee_model.get_return_value(exceedance_probability=0.1, bad_argument=1)
        with pytest.raises(ValueError, match=r"invalid shape.*exceedance_probability"):
            emcee_model.get_return_value(exceedance_probability=[[1, 2, 3], [1, 2, 3]])

        # Test scalar, no alpha
        rv, cil, ciu = emcee_model.get_return_value(exceedance_probability=0.1)
        assert np.isclose(
            rv, scipy.stats.genextreme.isf(0.1, c=0, loc=10, scale=2), rtol=0, atol=0.1
        )
        assert np.isnan(cil) and np.isnan(ciu)
        assert len(emcee_model.return_value_cache) == 1

        # Test scalar, with alpha
        rv, cil, ciu = emcee_model.get_return_value(
            exceedance_probability=0.1, alpha=0.95
        )
        assert np.isclose(
            rv, scipy.stats.genextreme.isf(0.1, c=0, loc=10, scale=2), rtol=0, atol=0.1
        )
        assert not (np.isnan(cil) or np.isnan(ciu))
        assert len(emcee_model.return_value_cache) == 2

        # Test array, with alpha
        rv, cil, ciu = emcee_model.get_return_value(
            exceedance_probability=[0.1, 0.2], alpha=0.95
        )
        assert np.allclose(
            rv,
            scipy.stats.genextreme.isf([0.1, 0.2], c=0, loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not np.any(np.isnan(cil) | np.isnan(ciu))
        assert len(emcee_model.return_value_cache) == 3

    def test_repr(self, emcee_model):
        repr_value = str(emcee_model)
        assert len(repr_value.split("\n")) == 10

    @pytest.mark.parametrize(
        "distribution_name, theta, distribution_kwargs, scipy_parameters",
        [
            ("genextreme", (0.5, 10, 2), {}, (0.5, 10, 2)),
            ("gumbel_r", (10, 2), {}, (10, 2)),
            ("genpareto", (0.5, 0, 2), {}, (0.5, 0, 2)),
            ("genpareto", (0.5, 2), {"floc": 0}, (0.5, 0, 2)),
            ("expon", (0, 2), {}, (0, 2)),
            ("expon", (2,), {"floc": 0}, (0, 2)),
        ],
    )
    def test_emcee_distributions(
        self, distribution_name, theta, distribution_kwargs, scipy_parameters
    ):
        scipy_distribution = getattr(scipy.stats, distribution_name)

        np.random.seed(0)
        extremes = pd.Series(
            index=pd.date_range(start="2000-01-01", periods=100, freq="1H"),
            data=scipy_distribution.rvs(*scipy_parameters, size=100),
        )
        model = get_model(
            model="Emcee",
            extremes=extremes,
            distribution=distribution_name,
            distribution_kwargs=distribution_kwargs,
            n_walkers=20,
            n_samples=100,
        )

        # Test extremes attribute
        assert np.all(model.extremes.index == extremes.index)
        assert np.allclose(model.extremes.values, extremes.values)

        # Test fit_parameters attribute
        assert model.distribution.name == distribution_name
        assert len(model.fit_parameters) == len(scipy_parameters) - len(
            distribution_kwargs
        )

        # Test emcee attributes
        assert model.n_walkers == 20
        assert model.n_samples == 100

        # Test trace attribute
        assert model.trace.shape == (20, 100, len(model.fit_parameters))
        assert np.issubdtype(model.trace.dtype, np.float64)

        # Test return_value_cache attribute
        assert isinstance(model.return_value_cache, dict)
        assert len(model.return_value_cache) == 0

        # Test name attribute
        assert model.name == "Emcee"

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
        assert len(repr_value.split("\n")) == 10
        if len(distribution_kwargs) == 0:
            assert "all parameters are free" in repr_value

        # Test get_return_value
        with pytest.raises(TypeError, match=r"unrecognized arguments.*bad_argument"):
            model.get_return_value(exceedance_probability=0.1, bad_argument=1)
        with pytest.raises(ValueError, match=r"invalid shape.*exceedance_probability"):
            model.get_return_value(exceedance_probability=[[1, 2, 3], [1, 2, 3]])

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
