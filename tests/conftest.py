import pathlib

import pandas as pd
import pytest

root_directory = pathlib.Path(__file__).parent


@pytest.fixture(scope="session")
def shared_data_folder() -> pathlib.Path:
    return root_directory / "data"


@pytest.fixture(scope="function")
def battery_wl(shared_data_folder) -> pd.Series:
    ts = (
        pd.read_csv(
            shared_data_folder / "battery_wl.csv",
            index_col=0,
            parse_dates=True,
            squeeze=True,
        )
        .dropna()
        .sort_index(ascending=True)
    )
    ts = ts.loc[ts.index.year >= 1925]
    ts = (
        ts
        - (ts.index.array - pd.to_datetime("1992-01-01"))
        / pd.to_timedelta("1Y")
        * 2.87e-3
    )
    return ts
