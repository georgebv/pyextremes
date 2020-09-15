import numpy as np
import pandas as pd
import pytest

from pyextremes import EVA, get_model


@pytest.fixture(scope="function")
def eva_model(battery_wl_preprocessed) -> EVA:
    return EVA(data=battery_wl_preprocessed)


class TestEVA:
    def test_init_errors(self):
        with pytest.raises(
            TypeError, match=r"invalid type.*'data' argument.*pandas.Series"
        ):
            EVA(data=1)

        with pytest.warns(RuntimeWarning, match=r"'data'.*not numeric.*converted"):
            eva_model = EVA(
                data=pd.Series(
                    data=["1", "2", "3"],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                )
            )
            assert np.allclose(eva_model.data.values, [1, 2, 3])

        with pytest.raises(TypeError, match=r"invalid dtype.*'data' argument.*numeric"):
            EVA(
                data=pd.Series(
                    data=["a", "b", "c"],
                    index=pd.DatetimeIndex(["2020", "2021", "2022"]),
                )
            )

        with pytest.raises(TypeError, match=r"index of 'data'.*date-time.*not"):
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
        assert eva_model.data.index.is_all_dates
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
                "block_size": "1Y",
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
            assert len(eva_model.extremes_kwargs) == 2
            assert eva_model.extremes_kwargs["block_size"] == pd.to_timedelta("1Y")
            assert eva_model.extremes_kwargs["errors"] == "raise"
        else:
            assert len(eva_model.extremes_kwargs) == 2
            assert eva_model.extremes_kwargs["threshold"] == 1.35
            assert eva_model.extremes_kwargs["r"] == pd.to_timedelta("24H")
        with pytest.raises(AttributeError, match=r"model must first"):
            eva_model.model

    @pytest.mark.parametrize(
        "extremes_params",
        [
            {"method": "BM", "block_size": "1Y"},
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
