import pathlib

import pytest


root_directory = pathlib.Path(__file__).parent


@pytest.fixture(scope="session")
def data_folder() -> pathlib.Path:
    return root_directory / "data"
