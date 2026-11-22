"""Tests for the `observed_duration` argument (POT rate on gappy data, #123)."""

import numpy as np
import pandas as pd
import pytest

from pyextremes import EVA, get_return_periods

THRESHOLD = 10.0
RATE_PER_YEAR = 5.0
SCALE = 1.0
YEAR = pd.to_timedelta("365.2425D")


def _synthetic_daily(n_years: int, seed: int) -> pd.Series:
    """
    Daily series with Poisson exceedances of THRESHOLD.

    Background values stay below the threshold. On average RATE_PER_YEAR days
    per year exceed it by an exponential excess with scale SCALE, so the
    true POT return level is THRESHOLD + SCALE * ln(RATE_PER_YEAR * T).
    """
    rng = np.random.default_rng(seed)
    n_days = round(n_years * 365.2425)
    values = rng.uniform(0.0, THRESHOLD - 1.0, size=n_days)
    events = rng.random(n_days) < RATE_PER_YEAR / 365.2425
    values[events] = THRESHOLD + rng.exponential(SCALE, size=events.sum())
    index = pd.date_range("1700-01-01", periods=n_days, freq="D")
    return pd.Series(values, index=index, name="synthetic")


def _with_gap(ts: pd.Series, gap_years: int) -> pd.Series:
    """Split `ts` in half and shift the second half later by `gap_years`."""
    half = len(ts) // 2
    second = ts.iloc[half:].copy()
    second.index = second.index + pd.to_timedelta(f"{gap_years * 365.2425}D")
    return pd.concat([ts.iloc[:half], second])


def _pot_model(ts: pd.Series, observed_duration=None) -> EVA:
    model = EVA(ts, observed_duration=observed_duration)
    model.get_extremes(method="POT", threshold=THRESHOLD, r="24h")
    model.fit_model(model="MLE", distribution="expon")
    return model


def test_observed_duration_validation():
    ts = _synthetic_daily(n_years=2, seed=0)
    assert EVA(ts).observed_duration is None
    assert EVA(ts, observed_duration="730D").observed_duration == pd.to_timedelta(
        "730D"
    )
    with pytest.raises(TypeError, match="observed_duration"):
        EVA(ts, observed_duration=730)
    with pytest.raises(ValueError, match="positive"):
        EVA(ts, observed_duration="0D")
    with pytest.raises(ValueError, match="positive"):
        EVA(ts, observed_duration="-1D")


def test_gap_with_observed_duration_matches_contiguous_record():
    """A gappy record with the right observed_duration equals the gap-free one."""
    contiguous = _synthetic_daily(n_years=40, seed=1)
    gappy = _with_gap(contiguous, gap_years=40)
    observed = contiguous.index[-1] - contiguous.index[0]

    model_ref = _pot_model(contiguous)
    model_fix = _pot_model(gappy, observed_duration=observed)
    model_default = _pot_model(gappy)

    # Same exceedances, so the fitted distribution is identical
    assert len(model_ref.extremes) == len(model_fix.extremes)
    np.testing.assert_allclose(
        model_ref.extremes.values, model_fix.extremes.values, rtol=0, atol=0
    )

    rp = [2, 10, 100]
    rv_ref = model_ref.get_return_value(rp)[0]
    rv_fix = model_fix.get_return_value(rp)[0]
    rv_default = model_default.get_return_value(rp)[0]
    np.testing.assert_allclose(rv_fix, rv_ref, rtol=1e-10)
    # Default uses the 80-year calendar span for 40 years of data: biased low
    assert np.all(rv_default < rv_ref - 0.5)

    # Empirical return periods agree too, through EVA and the standalone function
    obs_ref = get_return_periods(
        ts=contiguous,
        extremes=model_ref.extremes,
        extremes_method="POT",
        extremes_type="high",
    )
    obs_fix = get_return_periods(
        ts=gappy,
        extremes=model_fix.extremes,
        extremes_method="POT",
        extremes_type="high",
        observed_duration=observed,
    )
    np.testing.assert_allclose(
        obs_fix["return period"].values, obs_ref["return period"].values
    )


def test_gappy_record_recovers_known_rate_and_level():
    """With observed_duration, the event rate and 100-year level match truth."""
    # Keep the calendar span under ~292 years: pandas < 3 stores Timedelta in ns
    n_years, gap_years = 140, 140
    gappy = _with_gap(_synthetic_daily(n_years=n_years, seed=2), gap_years)
    observed = pd.to_timedelta(f"{n_years * 365.2425}D")
    true_rl = THRESHOLD + SCALE * np.log(RATE_PER_YEAR * 100)

    def implied_rate(model: EVA) -> float:
        # For an exponential excess, rv(T) = u + scale * ln(rate * T),
        # so the rate used by get_return_value can be read back exactly
        scale = model.model.fit_parameters["scale"]
        rv = model.get_return_value(100)[0]
        return float(np.exp((rv - THRESHOLD) / scale) / 100)

    model_fix = _pot_model(gappy, observed_duration=observed)
    model_default = _pot_model(gappy)

    # ~700 events: Poisson SD of the rate is ~0.19 per year
    assert abs(implied_rate(model_fix) - RATE_PER_YEAR) < 0.5
    # The default divides by the 280-year calendar span: half the true rate
    assert abs(implied_rate(model_default) - RATE_PER_YEAR / 2) < 0.25

    # Return level: sampling SD is ~0.25 here, so allow a generous tolerance
    rv_fix = model_fix.get_return_value(100)[0]
    rv_default = model_default.get_return_value(100)[0]
    assert abs(rv_fix - true_rl) < 0.6
    # Halving the rate lowers the level by scale * ln 2
    np.testing.assert_allclose(
        rv_fix - rv_default,
        model_fix.model.fit_parameters["scale"] * np.log(2),
        rtol=1e-4,  # calendar span is 280 years minus one day
    )


def test_observed_duration_ignored_by_block_maxima():
    ts = _synthetic_daily(n_years=30, seed=3)
    values = []
    for observed_duration in [None, "1000D"]:
        model = EVA(ts, observed_duration=observed_duration)
        model.get_extremes(method="BM", block_size="365.2425D")
        model.fit_model(model="MLE", distribution="gumbel_r")
        values.append(model.get_return_value([10, 100])[0])
    np.testing.assert_allclose(values[0], values[1])


def test_from_extremes_observed_duration():
    """from_extremes with POT: default span runs first-to-last extreme."""
    ts = _synthetic_daily(n_years=50, seed=4)
    record = ts.index[-1] - ts.index[0]
    extremes = _pot_model(ts).extremes

    rates = {}
    for label, od in [("default", None), ("record", record)]:
        model = EVA.from_extremes(
            extremes,
            method="POT",
            threshold=THRESHOLD,
            r="24h",
            observed_duration=od,
        )
        assert model.observed_duration == (None if od is None else record)
        model.fit_model(model="MLE", distribution="expon")
        rates[label] = model.get_return_value(100)[0]

    reference = _pot_model(ts).get_return_value(100)[0]
    np.testing.assert_allclose(rates["record"], reference, rtol=1e-10)
    # The first-to-last-extreme span is shorter than the record, so the
    # default rate is too high and the return value too high
    assert rates["default"] >= rates["record"]


def test_observed_duration_is_keyword_only():
    ts = _synthetic_daily(n_years=5, seed=5)
    extremes = _pot_model(ts).extremes
    with pytest.raises(TypeError):
        EVA(ts, "730D")
    with pytest.raises(TypeError):
        EVA.from_extremes(extremes, "POT", "high", "730D", threshold=THRESHOLD)


def test_get_return_periods_rejects_non_positive_observed_duration():
    ts = _synthetic_daily(n_years=5, seed=6)
    extremes = _pot_model(ts).extremes
    for bad in ["0D", "-1D"]:
        with pytest.raises(ValueError, match="observed_duration"):
            get_return_periods(
                ts=ts,
                extremes=extremes,
                extremes_method="POT",
                extremes_type="high",
                observed_duration=bad,
            )
