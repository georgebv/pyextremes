"""Tests for the `min_block_count` argument (partly observed blocks, #123)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pyextremes import EVA
from pyextremes.extremes import get_extremes
from pyextremes.extremes.block_maxima import NoDataBlockWarning

BLOCK = "365.2425D"


def _daily_gumbel(n_years: int, seed: int) -> pd.Series:
    """Daily iid standard Gumbel values: annual maxima are Gumbel(ln 365.24, 1)."""
    rng = np.random.default_rng(seed)
    n_days = round(n_years * 365.2425)
    index = pd.date_range("1950-01-01", periods=n_days, freq="D")
    return pd.Series(rng.gumbel(0.0, 1.0, size=n_days), index=index, name="x")


def _thin_alternate_years(ts: pd.Series, keep_every: int) -> pd.Series:
    """In every other block, keep only every `keep_every`-th day."""
    block = ((ts.index - ts.index[0]) / pd.to_timedelta(BLOCK)).astype(int)
    position = np.arange(len(ts))
    keep = (block % 2 == 0) | (position % keep_every == 0)
    return ts[keep]


def test_invalid_min_block_count():
    ts = _daily_gumbel(n_years=3, seed=0)
    with pytest.raises(TypeError, match="min_block_count"):
        get_extremes(ts, "BM", block_size=BLOCK, min_block_count=10.5)
    with pytest.raises(TypeError, match="min_block_count"):
        get_extremes(ts, "BM", block_size=BLOCK, min_block_count=True)
    with pytest.raises(ValueError, match="min_block_count"):
        get_extremes(ts, "BM", block_size=BLOCK, min_block_count=0)


def test_default_matches_previous_behaviour():
    ts = _thin_alternate_years(_daily_gumbel(n_years=10, seed=1), keep_every=10)
    baseline = get_extremes(ts, "BM", block_size=BLOCK, errors="ignore")
    for value in [None, 1, np.int64(1)]:
        extremes = get_extremes(
            ts, "BM", block_size=BLOCK, errors="ignore", min_block_count=value
        )
        pd.testing.assert_series_equal(extremes, baseline)


@pytest.mark.parametrize("errors", ["ignore", "coerce", "raise"])
def test_sparse_blocks_follow_errors(errors):
    ts = _thin_alternate_years(_daily_gumbel(n_years=10, seed=2), keep_every=10)
    kwargs = {"block_size": BLOCK, "errors": errors, "min_block_count": 300}

    if errors == "raise":
        with pytest.raises(ValueError, match="min_block_count=300"):
            get_extremes(ts, "BM", **kwargs)
        return

    with pytest.warns(NoDataBlockWarning, match="fewer than min_block_count=300"):
        extremes = get_extremes(ts, "BM", **kwargs)

    full = get_extremes(ts, "BM", block_size=BLOCK, errors="ignore")
    if errors == "ignore":
        # Only the five fully observed blocks are kept, with their true maxima
        assert len(extremes) == 5
        pd.testing.assert_series_equal(extremes, full.iloc[::2])
    else:
        # Sparse blocks are filled with the mean of the full blocks
        assert len(extremes) == 10
        np.testing.assert_allclose(extremes.iloc[::2].values, full.iloc[::2].values)
        np.testing.assert_allclose(
            extremes.iloc[1::2].values, full.iloc[::2].mean(), rtol=1e-12
        )


def test_ignored_last_block_does_not_drop_previous_block():
    """min_last_block must not discard the previous block's maximum."""
    ts = _daily_gumbel(n_years=5, seed=3)
    ts = ts[ts.index < ts.index[0] + pd.to_timedelta("1490D")]  # last block ~30 days
    with pytest.warns(NoDataBlockWarning):
        extremes = get_extremes(
            ts,
            "BM",
            block_size=BLOCK,
            errors="ignore",
            min_last_block=0.5,
            min_block_count=300,
        )
    full = get_extremes(ts, "BM", block_size=BLOCK, errors="ignore")
    assert len(extremes) == 4
    pd.testing.assert_series_equal(extremes, full.iloc[:4])


def test_recovers_known_gumbel_parameters_on_gappy_data():
    """Partly observed blocks bias BM low; min_block_count removes the bias."""
    ts = _thin_alternate_years(_daily_gumbel(n_years=120, seed=4), keep_every=10)
    true_loc = np.log(365.2425)

    fits = {}
    for label, min_block_count in [("default", None), ("fixed", 300)]:
        model = EVA(ts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDataBlockWarning)
            model.get_extremes(
                method="BM",
                block_size=BLOCK,
                errors="ignore",
                min_block_count=min_block_count,
            )
        assert model.extremes_kwargs["min_block_count"] == min_block_count
        model.fit_model(model="MLE", distribution="gumbel_r")
        fits[label] = (len(model.extremes), model.model.fit_parameters["loc"])

    n_default, loc_default = fits["default"]
    n_fixed, loc_fixed = fits["fixed"]
    assert n_default == 120
    assert n_fixed == 60
    # 60 maxima: SE of the Gumbel location is about 0.14
    assert abs(loc_fixed - true_loc) < 0.45
    # Thinned blocks have ~36 days, so their maxima sit near ln(36.5),
    # about ln(10) = 2.3 below the full blocks: the default fit is pulled down
    assert loc_default < true_loc - 0.6


def test_set_extremes_records_min_block_count():
    ts = _daily_gumbel(n_years=20, seed=5)
    extremes = get_extremes(ts, "BM", block_size=BLOCK)
    model = EVA(ts)
    model.set_extremes(extremes, method="BM", block_size=BLOCK, min_block_count=300)
    assert model.extremes_kwargs["min_block_count"] == 300
    for bad in [0, -5, 2.5, True]:
        with pytest.raises(ValueError, match="min_block_count"):
            model.set_extremes(
                extremes, method="BM", block_size=BLOCK, min_block_count=bad
            )
