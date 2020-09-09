import pathlib

import pandas as pd
import pytest

root_directory = pathlib.Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def shared_data_folder() -> pathlib.Path:
    return root_directory / "_shared_data"


@pytest.fixture(scope="function")
def battery_wl(shared_data_folder) -> pd.Series:
    return (
        pd.read_csv(
            shared_data_folder / "battery_wl.csv",
            index_col=0,
            parse_dates=True,
            squeeze=True,
        )
        .dropna()
        .sort_index(ascending=True)
    )


@pytest.fixture(scope="function")
def battery_wl_preprocessed(shared_data_folder) -> pd.Series:
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
    slr = (
        (ts.index.array - pd.to_datetime("1992-01-01"))
        / pd.to_timedelta("1Y")
        * 2.87e-3
    )
    ts -= slr
    return ts


@pytest.fixture(scope="function")
def extremes_bm_high(shared_data_folder) -> pd.Series:
    return pd.read_csv(
        shared_data_folder / "extremes_bm_high.csv",
        index_col=0,
        parse_dates=True,
        squeeze=True,
    )


@pytest.fixture(scope="function")
def extremes_bm_low(shared_data_folder) -> pd.Series:
    return pd.read_csv(
        shared_data_folder / "extremes_bm_high.csv",
        index_col=0,
        parse_dates=True,
        squeeze=True,
    )


@pytest.fixture(scope="function")
def extremes_pot_high(shared_data_folder) -> pd.Series:
    return pd.read_csv(
        shared_data_folder / "extremes_pot_high.csv",
        index_col=0,
        parse_dates=True,
        squeeze=True,
    )


@pytest.fixture(scope="function")
def extremes_pot_low(shared_data_folder) -> pd.Series:
    return pd.read_csv(
        shared_data_folder / "extremes_pot_low.csv",
        index_col=0,
        parse_dates=True,
        squeeze=True,
    )
