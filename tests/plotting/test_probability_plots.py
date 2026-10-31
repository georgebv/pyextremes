import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats

from pyextremes.plotting import plot_probability

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_probability_shows_coefficient_of_determination():
    np.random.seed(0)
    theoretical = np.sort(np.random.normal(size=50))
    observed = theoretical + np.random.normal(scale=0.5, size=50)
    pearson_r, p_value = scipy.stats.pearsonr(theoretical, observed)

    _, ax = plot_probability(observed=observed, theoretical=theoretical)

    (text,) = ax.texts
    assert text.get_text() == f"$R^2={pearson_r**2:.3f}$\n$p={p_value:.3f}$"
    assert not np.isclose(pearson_r, pearson_r**2, rtol=0, atol=1e-3)
