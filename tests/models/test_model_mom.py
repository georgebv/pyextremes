import os

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from pyextremes.models import MOM, get_model


@pytest.fixture(scope="function")
def extremes() -> pd.Series:
    np.random.seed(0)
    return pd.Series(
        index=pd.date_range(start="2000-01-01", periods=1000, freq="1h"),
        data=scipy.stats.gumbel_r.rvs(loc=10, scale=2, size=1000),
    )


@pytest.fixture(scope="function")
def mom_model(extremes) -> MOM:
    return get_model(
        model="MOM",
        extremes=extremes,
        distribution="gumbel_r",
    )


class Testmom:
    def test_model(self, extremes, mom_model):
        # Test extremes attribute
        assert np.all(mom_model.extremes.index == extremes.index)
        assert np.allclose(mom_model.extremes.values, extremes.values)

        # Test fit_parameters attribute
        assert mom_model.distribution.name == "gumbel_r"
        assert len(mom_model.fit_parameters) == 2
        for key, value in {"loc": 10, "scale": 2}.items():
            assert key in mom_model.fit_parameters
            assert np.isclose(mom_model.fit_parameters[key], value, rtol=0, atol=0.1)

        # Test trace attribute
        with pytest.raises(TypeError, match=r"trace property is not"):
            mom_model.trace

        # Test return_value_cache attribute
        assert isinstance(mom_model.return_value_cache, dict)
        assert len(mom_model.return_value_cache) == 0

        # Test name attribute
        assert mom_model.name == "MOM"

        # Test fit_parameter_cache attribute
        assert isinstance(mom_model.fit_parameter_cache, list)
        assert len(mom_model.fit_parameter_cache) == 0

    def test_fit(self, mom_model):
        with pytest.raises(TypeError, match=r"unrecognized arguments.*bad_argument"):
            mom_model.fit(bad_argument=1)

    def test_loglikelihood(self, extremes, mom_model):
        assert np.allclose(
            np.sum(
                scipy.stats.gumbel_r.logpdf(
                    x=extremes.values,
                    **mom_model.fit_parameters,
                    **mom_model.distribution._fixed_parameters,
                )
            ),
            mom_model.loglikelihood,
            rtol=0,
            atol=0.01,
        )

    def test_aic(self, extremes, mom_model):
        k = 2
        n = len(extremes)
        aic = 2 * k - 2 * mom_model.loglikelihood
        correction = (2 * k**2 + 2 * k) / (n - k - 1)
        assert np.isclose(mom_model.AIC, aic + correction, rtol=0, atol=3)

    @pytest.mark.parametrize("prop", ["pdf", "cdf", "ppf", "isf"])
    def test_properties(self, prop, mom_model):
        assert np.isclose(
            getattr(mom_model, prop)(x=0.1),
            getattr(scipy.stats.gumbel_r, prop)(0.1, loc=10, scale=2),
            rtol=0.1,
            atol=0.2,
        )

    def test_get_return_value(self, mom_model):
        with pytest.raises(TypeError, match=r"unrecognized arguments.*bad_argument"):
            mom_model.get_return_value(exceedance_probability=0.1, bad_argument=1)
        with pytest.raises(ValueError, match=r"invalid shape.*exceedance_probability"):
            mom_model.get_return_value(exceedance_probability=[[1, 2, 3], [1, 2, 3]])
        with pytest.raises(
            ValueError, match=r"invalid value.*n_samples.*must be positive"
        ):
            mom_model.get_return_value(exceedance_probability=0.1, n_samples=-1)

        # Test scalar, no alpha
        rv, cil, ciu = mom_model.get_return_value(exceedance_probability=0.1)
        assert np.isclose(
            rv, scipy.stats.gumbel_r.isf(0.1, loc=10, scale=2), rtol=0, atol=0.1
        )
        assert np.isnan(cil) and np.isnan(ciu)
        assert len(mom_model.return_value_cache) == 1
        assert len(mom_model.fit_parameter_cache) == 0
        assert len(np.unique(mom_model.fit_parameter_cache, axis=0)) == len(
            mom_model.fit_parameter_cache
        )
        seed_cahe_size = 0
        assert len(mom_model.seed_cache) == seed_cahe_size

        # Test scalar, with alpha
        rv, cil, ciu = mom_model.get_return_value(
            exceedance_probability=0.1, alpha=0.95
        )
        assert np.isclose(
            rv,
            scipy.stats.gumbel_r.isf(0.1, loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not (np.isnan(cil) or np.isnan(ciu))
        assert len(mom_model.return_value_cache) == 2
        assert len(mom_model.fit_parameter_cache) == 100
        assert len(np.unique(mom_model.fit_parameter_cache, axis=0)) == len(
            mom_model.fit_parameter_cache
        )
        seed_cahe_size = min(2, os.cpu_count())
        assert len(mom_model.seed_cache) == seed_cahe_size

        # Test array, with alpha
        rv, cil, ciu = mom_model.get_return_value(
            exceedance_probability=[0.1, 0.2], alpha=0.95
        )
        assert np.allclose(
            rv,
            scipy.stats.gumbel_r.isf([0.1, 0.2], loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not np.any(np.isnan(cil) | np.isnan(ciu))
        assert len(mom_model.return_value_cache) == 3
        assert len(mom_model.fit_parameter_cache) == 100
        assert len(np.unique(mom_model.fit_parameter_cache, axis=0)) == len(
            mom_model.fit_parameter_cache
        )
        seed_cahe_size = min(2, os.cpu_count())
        assert len(mom_model.seed_cache) == seed_cahe_size

        # Test small additional n_samples
        rv, cil, ciu = mom_model.get_return_value(
            exceedance_probability=0.1,
            alpha=0.95,
            n_samples=120,
        )
        assert np.isclose(
            rv,
            scipy.stats.gumbel_r.isf(0.1, loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not (np.isnan(cil) or np.isnan(ciu))
        assert len(mom_model.return_value_cache) == 4
        assert len(np.unique(mom_model.fit_parameter_cache, axis=0)) == len(
            mom_model.fit_parameter_cache
        )
        seed_cahe_size = min(2, os.cpu_count()) + 1
        assert len(mom_model.seed_cache) == seed_cahe_size

        # Test large additional n_samples, not multiple of 50
        rv, cil, ciu = mom_model.get_return_value(
            exceedance_probability=0.1,
            alpha=0.95,
            n_samples=201,
        )
        assert np.isclose(
            rv,
            scipy.stats.gumbel_r.isf(0.1, loc=10, scale=2),
            rtol=0,
            atol=0.1,
        )
        assert not (np.isnan(cil) or np.isnan(ciu))
        assert len(mom_model.return_value_cache) == 5
        assert len(mom_model.fit_parameter_cache) == 201
        assert len(np.unique(mom_model.fit_parameter_cache, axis=0)) == len(
            mom_model.fit_parameter_cache
        )
        seed_cahe_size += min(2, os.cpu_count())
        assert len(mom_model.seed_cache) == seed_cahe_size

    def test_repr(self, mom_model):
        repr_value = str(mom_model)
        assert len(repr_value.split("\n")) == 9

    @pytest.mark.parametrize(
        "distribution_name, distribution_kwargs, scipy_parameters",
        [
            ("gumbel_r", {}, (10, 2)),
            ("expon", {}, (0, 2)),
            ("expon", {}, (0, 2)),
        ],
    )
    def test_mom_distributions(
        self, distribution_name, distribution_kwargs, scipy_parameters
    ):
        scipy_distribution = getattr(scipy.stats, distribution_name)

        np.random.seed(0)
        extremes = pd.Series(
            index=pd.date_range(start="2000-01-01", periods=100, freq="1h"),
            data=scipy_distribution.rvs(*scipy_parameters, size=100),
        )
        model = get_model(
            model="MOM",
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
        assert model.name == "MOM"

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
        correction = (2 * k**2 + 2 * k) / (n - k - 1)
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
