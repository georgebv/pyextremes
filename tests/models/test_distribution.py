import numpy as np
import pandas as pd
import pytest
import scipy.stats

from pyextremes.models.distribution import Distribution


class TestDistribution:
    def test_errors(self):
        # Test bad distribution type
        with pytest.raises(TypeError, match=r"invalid type.*'distribution' argument"):
            Distribution(extremes=pd.Series([1, 2, 3]), distribution=1)

        # Test discrete distribution
        with pytest.raises(ValueError, match=r"continuous distribution"):
            Distribution(extremes=pd.Series([1, 2, 3]), distribution="poisson")

        # Test bad kwargs
        with pytest.raises(TypeError, match=r"not a valid keyword.*distribution"):
            Distribution(extremes=pd.Series([1, 2, 3]), distribution="expon", fc=0)

        # Test all parameters being fixed
        with pytest.raises(ValueError, match=r"all parameters.*fixed.*nothing to fit"):
            Distribution(
                extremes=pd.Series([1, 2, 3]),
                distribution="genextreme",
                fc=0,
                floc=0,
                fscale=1,
            )

    @pytest.mark.parametrize(
        "distribution_input, theta, kwargs, scipy_parameters",
        [
            ("genextreme", (0.5, 10, 2), {}, (0.5, 10, 2)),
            (scipy.stats.genextreme, (0.5, 10, 2), {}, (0.5, 10, 2)),
            ("gumbel_r", (10, 2), {}, (10, 2)),
            ("genpareto", (0.5, 0, 2), {}, (0.5, 0, 2)),
            ("genpareto", (0.5, 2), {"floc": 0}, (0.5, 0, 2)),
            ("expon", (0, 2,), {}, (0, 2)),
            ("expon", (2,), {"floc": 0}, (0, 2)),
        ],
    )
    def test_distribution(self, distribution_input, theta, kwargs, scipy_parameters):
        if isinstance(distribution_input, scipy.stats.rv_continuous):
            scipy_distribution = distribution_input
            distribution_name = distribution_input.name
        else:
            distribution_name = distribution_input
            scipy_distribution = getattr(scipy.stats, distribution_name)
        distribution = Distribution(
            extremes=pd.Series(
                index=pd.date_range(start="2000-01-01", periods=100, freq="1H"),
                data=scipy_distribution.rvs(*scipy_parameters, size=100),
            ),
            distribution=distribution_input,
            **kwargs
        )

        # Test '__init__' and 'fit' methods
        assert distribution.distribution.name == distribution_name
        assert distribution.distribution.badvalue == -np.inf
        if scipy_distribution.shapes is not None:
            assert (
                len(distribution.distribution_parameters)
                == len(scipy_distribution.shapes) + 2
            )
        assert len(distribution.distribution_parameters) == len(scipy_parameters)
        assert distribution.fixed_parameters == kwargs
        assert len(distribution.fixed_parameters) == len(kwargs)
        for key, value in distribution.fixed_parameters.items():
            assert key in kwargs
            assert value == kwargs[key]
        assert len(distribution.free_parameters) == (
            len(distribution.distribution_parameters) - len(kwargs)
        )
        for key in distribution.mle_parameters.keys():
            assert key in distribution.free_parameters
            assert key not in distribution._fixed_parameters
        assert len(distribution.mle_parameters) == distribution.number_of_parameters

        # Test properties
        assert distribution.name == distribution_name
        assert distribution.number_of_parameters == len(
            distribution.distribution_parameters
        ) - len(kwargs)

        # Test repr
        repr_value = str(distribution)
        assert len(repr_value.split("\n")) == 7

        # Test 'log_probability' method
        assert np.isclose(
            distribution.log_probability(theta=theta),
            sum(
                scipy_distribution.logpdf(
                    distribution.extremes.values, *scipy_parameters
                )
            ),
        )
        with pytest.raises(ValueError, match=r"invalid theta.*must have size"):
            distribution.log_probability(theta=list(range(100)))

        # Test 'get_initial_state' method
        initial_state = distribution.get_initial_state(n_walkers=1000)
        assert initial_state.shape == (1000, distribution.number_of_parameters)
        assert np.allclose(
            initial_state.mean(axis=0),
            list(distribution.mle_parameters.values()),
            atol=0.1,
        )

        # Test 'free2full_parameters' method
        theta_dict = {
            parameter: theta[i]
            for i, parameter in enumerate(distribution.free_parameters)
        }
        ffp = distribution.free2full_parameters(free_parameters=theta_dict)
        assert ffp.shape == (len(scipy_parameters),)
        assert np.allclose(ffp, scipy_parameters)

        ffp = distribution.free2full_parameters(free_parameters=theta)
        assert np.allclose(ffp, scipy_parameters)
        assert ffp.shape == (len(scipy_parameters),)

        ffp = distribution.free2full_parameters(free_parameters=[theta] * 5)
        assert ffp.shape == (5, len(scipy_parameters),)
        assert np.all([np.allclose(row, scipy_parameters) for row in ffp])

        if len(theta) == 1:
            ffp = distribution.free2full_parameters(free_parameters=theta[0])
            assert np.allclose(ffp, scipy_parameters)
            assert ffp.shape == (len(scipy_parameters),)

        with pytest.raises(
            ValueError,
            match=r"invalid value[\s\S]*'free_parameters' argument.*must have length",
        ):
            distribution.free2full_parameters(
                free_parameters={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
            )

        with pytest.raises(
            ValueError,
            match=r"invalid value[\s\S]*'free_parameters' argument.*must have length",
        ):
            distribution.free2full_parameters(free_parameters=list(range(100)))

        with pytest.raises(
            ValueError, match=r"invalid shape.*'free_parameters' argument.*must be \(n",
        ):
            distribution.free2full_parameters(
                free_parameters=[list(range(100)), list(range(100))]
            )

        with pytest.raises(
            ValueError,
            match=r"invalid shape.*'free_parameters' argument.*must be 1D or 2D array",
        ):
            distribution.free2full_parameters(
                free_parameters=[[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]
            )

        # Test 'get_prop' method
        with pytest.raises(
            ValueError, match=r"invalid shape.*'x' argument.*must be 1D array"
        ):
            distribution.get_prop(
                prop="pdf", x=[[1, 2, 3], [1, 2, 3]], free_parameters=theta_dict
            )

        for prop in ["pdf", "cdf", "ppf", "isf"]:
            # Scalar x, 1D free_parameters
            assert np.isclose(
                distribution.get_prop(prop=prop, x=0.1, free_parameters=theta_dict),
                getattr(scipy_distribution, prop)(0.1, *scipy_parameters),
            )

            # 1D x, 1D free_parameters
            assert np.allclose(
                distribution.get_prop(
                    prop=prop, x=[0.1, 0.2], free_parameters=theta_dict
                ),
                getattr(scipy_distribution, prop)([0.1, 0.2], *scipy_parameters),
            )

            # Scalar x, 2D free_parameters
            assert np.allclose(
                distribution.get_prop(prop=prop, x=0.1, free_parameters=[theta, theta]),
                getattr(scipy_distribution, prop)(
                    0.1, *np.transpose([scipy_parameters, scipy_parameters])
                ),
            )

            # 2D x, 2D free_parameters
            assert np.allclose(
                distribution.get_prop(
                    prop=prop, x=[0.1, 0.2], free_parameters=[theta, theta]
                ),
                np.transpose(
                    getattr(scipy_distribution, prop)(
                        np.transpose([0.1, 0.2]),
                        *np.transpose(
                            [
                                [scipy_parameters, scipy_parameters],
                                [scipy_parameters, scipy_parameters],
                            ]
                        )
                    )
                ),
            )
