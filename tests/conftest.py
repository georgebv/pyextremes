import pathlib
import typing

import pytest


@pytest.fixture(scope="session")
def data_folder() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def temporary_folder() -> typing.Generator[pathlib.Path, None, None]:
    folder = pathlib.Path(__file__).parent / "__temporary_folder__"
    folder.mkdir()
    yield folder
    folder.rmdir()
