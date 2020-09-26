import numpy as np
import pytest
import scipy.stats
from pyextremes.tests import KolmogorovSmirnov


class TestKolmogorovSmirnov:
    def test_init_errors(self):
        with pytest.raises(TypeError, match=r"invalid type.*rv_continuous"):
            KolmogorovSmirnov(
                extremes=[1, 2, 3],
                distribution=1,
                fit_parameters={"a": 1},
            )

    @pytest.mark.parametrize(
        "distribution,fit_parameters",
        [
            [
                "genextreme",
                {"c": 0.3, "loc": 10, "scale": 2},
            ],
            [
                scipy.stats.genpareto,
                {"c": 0.3, "loc": 10, "scale": 2},
            ],
        ],
    )
    def test_init(self, distribution, fit_parameters):
        if isinstance(distribution, str):
            scipy_distribution = getattr(scipy.stats, distribution)
        else:
            scipy_distribution = distribution

        np.random.seed(12345)
        extremes = scipy_distribution.rvs(size=100, **fit_parameters)
        scipy_kstest = scipy.stats.kstest(
            rvs=extremes,
            cdf=lambda x: scipy_distribution.cdf(x, **fit_parameters),
        )

        kstest = KolmogorovSmirnov(
            extremes=extremes,
            distribution=distribution,
            fit_parameters=fit_parameters,
            significance_level=0.05,
        )

        assert kstest.name == "Kolmogorov-Smirnov"
        assert scipy_distribution.name in kstest.null_hypothesis
        assert scipy_distribution.name in kstest.alternative_hypothesis
        assert kstest.success

        assert np.isclose(kstest.test_statistic, scipy_kstest.statistic)
        assert np.isclose(kstest.pvalue, scipy_kstest.pvalue)
        assert np.isclose(
            kstest.critical_value,
            scipy.stats.ksone.ppf(1 - 0.05 / 2, len(extremes)),
        )

        # Test failure
        kstest = KolmogorovSmirnov(
            extremes=extremes,
            distribution=scipy.stats.norm,
            fit_parameters={"loc": 0, "scale": 1},
            significance_level=0.05,
        )
        assert not kstest.success
