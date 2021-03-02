import numpy as np
import pandas as pd
import pytest
import scipy.stats

from pyextremes import EVA, get_model


@pytest.fixture(scope="function")
def eva_model(battery_wl_preprocessed) -> EVA:
    return EVA(data=battery_wl_preprocessed)


@pytest.fixture(scope="function")
def eva_model_bm(battery_wl_preprocessed) -> EVA:
    eva_model = EVA(data=battery_wl_preprocessed)
    eva_model.get_extremes(
        method="BM",
        extremes_type="high",
        block_size="365.2425D",
        errors="raise",
    )
    return eva_model


@pytest.fixture(scope="function")
def eva_model_pot(battery_wl_preprocessed) -> EVA:
    eva_model = EVA(data=battery_wl_preprocessed)
    eva_model.get_extremes(
        method="POT",
        extremes_type="high",
        threshold=1.35,
        r="24H",
    )
    return eva_model


@pytest.fixture(scope="function")
def eva_model_bm_mle(battery_wl_preprocessed) -> EVA:
    eva_model = EVA(data=battery_wl_preprocessed)
    eva_model.get_extremes(
        method="BM",
        extremes_type="high",
        block_size="365.2425D",
        errors="raise",
    )
    eva_model.fit_model("MLE")
    return eva_model


@pytest.fixture(scope="function")
def eva_model_bm_emcee(battery_wl_preprocessed) -> EVA:
    eva_model = EVA(data=battery_wl_preprocessed)
    eva_model.get_extremes(
        method="BM",
        extremes_type="high",
        block_size="365.2425D",
        errors="raise",
    )
    eva_model.fit_model("Emcee", n_walkers=10, n_samples=100)
    return eva_model


@pytest.fixture(scope="function")
def eva_model_pot_mle(battery_wl_preprocessed) -> EVA:
    eva_model = EVA(data=battery_wl_preprocessed)
    eva_model.get_extremes(
        method="POT",
        extremes_type="high",
        threshold=1.35,
        r="24H",
    )
    eva_model.fit_model("MLE")
    return eva_model


class TestEVA:
    def test_init_errors(self):
        with pytest.raises(
            TypeError, match=r"invalid type.*`data` argument.*pandas.Series"
        ):
            EVA(data=1)

        with pytest.warns(RuntimeWarning, match=r"`data`.*not numeric.*converting"):
            eva_model = EVA(
                data=pd.Series(
                    data=["1", "2", "3"],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                )
            )
            assert np.allclose(eva_model.data.values, [1, 2, 3])

        with pytest.raises(TypeError, match=r"invalid dtype.*`data` argument.*numeric"):
            EVA(
                data=pd.Series(
                    data=["a", "b", "c"],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                )
            )

        with pytest.raises(TypeError, match=r"index of `data`.*date-time.*not"):
            EVA(data=pd.Series(data=[1, 2, 3], index=["2020", "2021", "2022"]))

        with pytest.warns(RuntimeWarning, match=r"index is not sorted.*sorting"):
            eva_model = EVA(
                data=pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2022", "2021", "2020"]),
                )
            )
            assert np.allclose(eva_model.data.index.year.values, [2020, 2021, 2022])

        with pytest.warns(RuntimeWarning, match=r"Null values found.*removing invalid"):
            eva_model = EVA(
                data=pd.Series(
                    data=[1, 2, np.nan, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022", "2023"]),
                )
            )
            assert np.allclose(eva_model.data.values, [1, 2, 3])
            assert np.allclose(eva_model.data.index.year.values, [2020, 2021, 2023])

    def test_init_attributes(self, eva_model):
        # Ensure that 'data' attribute is properly processed
        assert isinstance(eva_model.data, pd.Series)
        assert np.issubdtype(eva_model.data.dtype, np.number)
        assert isinstance(eva_model.data.index, pd.DatetimeIndex)
        assert eva_model.data.index.is_monotonic_increasing
        assert eva_model.data.isna().sum() == 0

        # Ensure model attributes exist and are all None
        for param in [
            "extremes",
            "extremes_method",
            "extremes_type",
            "extremes_kwargs",
            "extremes_transformer",
        ]:
            with pytest.raises(AttributeError, match=r"extreme values must first"):
                getattr(eva_model, param)

        with pytest.raises(AttributeError, match=r"model must first"):
            eva_model.model

    @pytest.mark.parametrize(
        "input_params",
        [
            {
                "method": "BM",
                "extremes_type": "high",
                "block_size": "365.2425D",
                "errors": "raise",
            },
            {
                "method": "BM",
                "extremes_type": "high",
            },
            {
                "method": "POT",
                "extremes_type": "high",
                "threshold": 1.35,
                "r": "24H",
            },
            {
                "method": "POT",
                "extremes_type": "high",
                "threshold": 1.35,
            },
        ],
    )
    def test_get_extremes(self, eva_model, input_params):
        # Get extremes
        eva_model.get_extremes(**input_params)

        # Test attributes
        assert eva_model.extremes_method == input_params["method"]
        assert eva_model.extremes_type == input_params["extremes_type"]
        if input_params["method"] == "BM":
            assert len(eva_model.extremes_kwargs) == 3
            assert eva_model.extremes_kwargs["block_size"] == pd.to_timedelta(
                "365.2425D"
            )
            assert eva_model.extremes_kwargs["errors"] == "raise"
        else:
            assert len(eva_model.extremes_kwargs) == 2
            assert eva_model.extremes_kwargs["threshold"] == 1.35
            assert eva_model.extremes_kwargs["r"] == pd.to_timedelta("24H")
        with pytest.raises(AttributeError, match=r"model must first"):
            eva_model.model

    def test_set_extremes_errors(self):
        eva_model = EVA(
            data=pd.Series(
                data=np.arange(100),
                index=pd.date_range(start="2000", end="2050", periods=100),
                name="water level [m]",
            )
        )

        # Test invalid `extremes`
        with pytest.raises(TypeError, match=r"invalid type.*must be pandas.Series"):
            eva_model.set_extremes([1, 2, 3])
        with pytest.raises(TypeError, match=r"invalid index.*must be date-time"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=[1, 2, 3],
                )
            )
        with pytest.raises(TypeError, match=r"`extremes` must have numeric values"):
            eva_model.set_extremes(
                pd.Series(
                    data=["a", "b", "c"],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                )
            )
        with pytest.raises(ValueError, match="name doesn't match"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name="different name",
                )
            )
        with pytest.raises(ValueError, match=".+time range must fit within.+"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["1990", "2021", "2022"]),
                )
            )

        # Test invalid general kwargs
        with pytest.raises(ValueError, match=r"`method` must be either.+"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name=eva_model.data.name,
                ),
                method="wrong method",
            )
        with pytest.raises(ValueError, match=r"`extremes_type` must be either.+"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name=eva_model.data.name,
                ),
                method="BM",
                extremes_type="wrong type",
            )

        # Test invalid BM kwargs
        with pytest.raises(ValueError, match=r"`block_size` must be a positive.+"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name=eva_model.data.name,
                ),
                method="BM",
                extremes_type="high",
                block_size="-1D",
            )
        with pytest.raises(ValueError, match=r"invalid value.+`errors` argument"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name=eva_model.data.name,
                ),
                method="BM",
                extremes_type="high",
                errors="wrong errors",
            )
        with pytest.raises(ValueError, match=r"`min_last_block` must be a number.+"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name=eva_model.data.name,
                ),
                method="BM",
                extremes_type="high",
                min_last_block=2.0,
            )

        # Test invalid POT kwargs
        with pytest.raises(ValueError, match=r"invalid `threshold` value"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name=eva_model.data.name,
                ),
                method="POT",
                extremes_type="high",
                threshold=2,
            )
        with pytest.raises(ValueError, match=r"`r` must be a positive.+"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name=eva_model.data.name,
                ),
                method="POT",
                extremes_type="high",
                r="-1D",
            )

        # Test unrecognized arguments
        with pytest.raises(TypeError, match=r"unrecognized arguments.+"):
            eva_model.set_extremes(
                pd.Series(
                    data=[1, 2, 3],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                    name=eva_model.data.name,
                ),
                method="BM",
                extremes_type="high",
                unrecognized_argument=1,
            )

    def test_from_extremes(self):
        index = pd.date_range(start="2000", end="2050", periods=100)
        eva_model = EVA.from_extremes(
            extremes=pd.Series(
                data=np.arange(100),
                index=index,
                name="water level [m]",
            ),
            method="BM",
            extremes_type="high",
        )
        assert eva_model.extremes_method == "BM"
        assert eva_model.extremes_type == "high"
        assert eva_model.extremes_kwargs["errors"] == "ignore"
        assert eva_model.extremes_kwargs["min_last_block"] is None
        expected_block_size = (
            ((index.max() - index.min()) / (len(index) - 1)).to_numpy().astype(float)
        )
        actual_block_size = (
            eva_model.extremes_kwargs["block_size"].to_numpy().astype(float)
        )
        assert np.isclose(expected_block_size, actual_block_size, rtol=0, atol=1e-6)

        # Test default POT arguments
        eva_model = EVA.from_extremes(
            extremes=pd.Series(
                data=np.arange(100),
                index=pd.date_range(start="2000", end="2050", periods=100),
                name="water level [m]",
            ),
            method="POT",
            extremes_type="high",
        )
        assert np.isclose(eva_model.extremes_kwargs["threshold"], 0, rtol=0, atol=1e-6)
        assert eva_model.extremes_kwargs["r"] == pd.to_timedelta("24H")
        eva_model = EVA.from_extremes(
            extremes=pd.Series(
                data=np.arange(100),
                index=pd.date_range(start="2000", end="2050", periods=100),
                name="water level [m]",
            ),
            method="POT",
            extremes_type="low",
        )
        assert np.isclose(eva_model.extremes_kwargs["threshold"], 99, rtol=0, atol=1e-6)
        assert eva_model.extremes_kwargs["r"] == pd.to_timedelta("24H")

    @pytest.mark.parametrize(
        "extremes_params",
        [
            {"method": "BM", "block_size": "365.2425D"},
            {"method": "POT", "threshold": 1.35},
        ],
    )
    def test_fit_model_default_distribution(self, eva_model, extremes_params):
        eva_model.get_extremes(**extremes_params)
        eva_model.fit_model()
        distributions = {
            "BM": ["genextreme", "gumbel_r"],
            "POT": ["genpareto", "expon"],
        }[extremes_params["method"]]
        distribution_kwargs = {
            "BM": {},
            "POT": {"floc": 1.35},
        }[extremes_params["method"]]
        assert eva_model.distribution.name in distributions
        aic = [
            get_model(
                model="MLE",
                extremes=eva_model.extremes,
                distribution=distribution,
                distribution_kwargs=distribution_kwargs,
            ).AIC
            for distribution in distributions
        ]
        assert np.isclose(eva_model.AIC, min(aic))

    def test_fit_model_errors(self, eva_model_bm, eva_model_pot):
        # Bad distribution type
        with pytest.raises(
            TypeError, match=r"invalid type.*distribution.*rv_continuous"
        ):
            eva_model_bm.fit_model(distribution=scipy.stats.poisson)

        # Distribution without name
        distribution = scipy.stats.genextreme.__class__()
        distribution.name = None
        with pytest.warns(RuntimeWarning, match=r"provided.*'name'.*validity"):
            eva_model_bm.fit_model(distribution=distribution)

        # Non-recommended distribution
        with pytest.warns(RuntimeWarning, match=r"is not recommended.*Fisher"):
            eva_model_bm.fit_model(distribution="genpareto")
            assert eva_model_bm.distribution.name == "genpareto"
        with pytest.warns(RuntimeWarning, match=r"is not recommended.*Pickands"):
            eva_model_pot.fit_model(distribution="genextreme")
            assert eva_model_pot.distribution.name == "genextreme"

    @pytest.mark.parametrize("model", ["MLE", "Emcee"])
    @pytest.mark.parametrize(
        "input_parameters",
        [
            {
                "extremes_method": "BM",
                "distribution": scipy.stats.genextreme,
                "distribution_kwargs": None,
            },
            {
                "extremes_method": "BM",
                "distribution": scipy.stats.genextreme,
                "distribution_kwargs": {"floc": 0},
            },
            {
                "extremes_method": "POT",
                "distribution": scipy.stats.genpareto,
                "distribution_kwargs": None,
            },
            {
                "extremes_method": "POT",
                "distribution": scipy.stats.genpareto,
                "distribution_kwargs": {"floc": 1.0},
            },
        ],
    )
    def test_fit_model(self, eva_model_bm, eva_model_pot, model, input_parameters):
        eva_model = {
            "BM": eva_model_bm,
            "POT": eva_model_pot,
        }[input_parameters["extremes_method"]]
        model_kwargs = {}
        if model == "Emcee":
            model_kwargs["n_walkers"] = 10
            model_kwargs["n_samples"] = 100
        eva_model.fit_model(
            model=model,
            distribution=input_parameters["distribution"],
            distribution_kwargs=input_parameters["distribution_kwargs"],
            **model_kwargs
        )

        assert eva_model.model.name == model
        assert eva_model.distribution.name == input_parameters["distribution"].name
        if eva_model.distribution.name == "genextreme":
            if input_parameters["distribution_kwargs"] is None:
                assert len(eva_model.distribution.fixed_parameters) == 0
            else:
                assert len(eva_model.distribution.fixed_parameters) == len(
                    input_parameters["distribution_kwargs"]
                )
                for key, value in input_parameters["distribution_kwargs"].items():
                    assert value == eva_model.distribution.fixed_parameters[key]
        elif eva_model.distribution.name == "genpareto":
            if input_parameters["distribution_kwargs"] is None:
                assert len(eva_model.distribution.fixed_parameters) == 1
                assert (
                    eva_model.distribution.fixed_parameters["floc"]
                    == eva_model.extremes_kwargs["threshold"]
                )
            else:
                assert len(eva_model.distribution.fixed_parameters) == 1
                assert (
                    eva_model.distribution.fixed_parameters["floc"]
                    == input_parameters["distribution_kwargs"]["floc"]
                )

    def test_get_mcmc_plot_inputs(self, eva_model_bm_mle, eva_model_bm_emcee):
        # Test trying to plot MCMC-only plots for MLE model
        with pytest.raises(TypeError, match=r"this method.*MCMC-like"):
            eva_model_bm_mle.plot_trace()
        with pytest.raises(TypeError, match=r"this method.*MCMC-like"):
            eva_model_bm_mle.plot_corner()

        # Test default behaviour
        trace, trace_map, labels = eva_model_bm_emcee._get_mcmc_plot_inputs()
        assert np.allclose(trace, eva_model_bm_emcee.model.trace)
        assert len(trace_map) == len(eva_model_bm_emcee.distribution.free_parameters)
        assert labels == [r"Shape, $\xi$", r"Location, $\mu$", r"Scale, $\sigma$"]

    @pytest.mark.parametrize("extremes_method", ["BM", "POT"])
    def test_get_return_value(
        self, eva_model_bm_mle, eva_model_pot_mle, extremes_method
    ):
        eva_model = {
            "BM": eva_model_bm_mle,
            "POT": eva_model_pot_mle,
        }[extremes_method]

        # Test invalid 'return_period_size' type
        with pytest.raises(TypeError, match=r"invalid type.*return_period_size"):
            eva_model.get_return_value(return_period=1, return_period_size=1)

        # Test invalid 'return_period' shape
        with pytest.raises(
            ValueError, match=r"invalid shape.*'return_period' argument"
        ):
            eva_model.get_return_value(return_period=[[1, 2], [3, 4]])

        # Test scalar outputs
        rv, cil, ciu = eva_model.get_return_value(
            return_period=100, return_period_size="365.2425D", alpha=0.95
        )
        assert all(isinstance(value, float) for value in (rv, cil, ciu))
        assert cil < rv < ciu
        assert np.allclose(eva_model.extremes.max(), rv, rtol=0, atol=2)

        # Test array-like outputs
        rv, cil, ciu = eva_model.get_return_value(
            return_period=[10, 100], return_period_size="365.2425D", alpha=0.95
        )
        for value in (rv, cil, ciu):
            assert isinstance(value, np.ndarray)
            assert value[1] > value[0]
        assert np.all((cil < rv) & (rv < ciu))

    @pytest.mark.parametrize("extremes_method", ["BM", "POT"])
    def test_get_summary(self, eva_model_bm_mle, eva_model_pot_mle, extremes_method):
        eva_model = {
            "BM": eva_model_bm_mle,
            "POT": eva_model_pot_mle,
        }[extremes_method]

        # Test invalid 'return_period' shape
        with pytest.raises(
            ValueError, match=r"invalid shape.*'return_period' argument"
        ):
            eva_model.get_summary(return_period=[[1, 2], [3, 4]])

        # Test scalar outputs
        rv_summary = eva_model.get_summary(
            return_period=100, return_period_size="365.2425D", alpha=0.95
        )
        assert isinstance(rv_summary, pd.DataFrame)
        assert len(rv_summary) == 1

        # Test array-like outputs
        rv_summary = eva_model.get_summary(
            return_period=[10, 100], return_period_size="365.2425D", alpha=0.95
        )
        assert isinstance(rv_summary, pd.DataFrame)
        assert len(rv_summary) == 2
