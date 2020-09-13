import numpy as np
import pandas as pd
import pytest

from pyextremes import EVA


@pytest.fixture(scope="function")
def eva_model(battery_wl_preprocessed) -> EVA:
    return EVA(data=battery_wl_preprocessed)


class TestEVA:
    def test_init_errors(self):
        with pytest.raises(
            TypeError, match=r"invalid type.*'data' argument.*pandas.Series"
        ):
            EVA(data=1)

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

        with pytest.warns(RuntimeWarning, match=r"nan values found.*removing invalid"):
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
            "model",
        ]:
            assert getattr(eva_model, param) is None
