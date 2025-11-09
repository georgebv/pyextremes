import logging
import typing

import numpy as np
import pandas as pd
import scipy.stats

try:
    from lmoments3 import distr as lmoments3_distr
except ModuleNotFoundError:  # pragma: no cover
    lmoments3_distr = None

logger = logging.getLogger(__name__)

# Maps scipy distribution names to lmoments3 distribution names
_SCIPY_TO_LMOMENTS3: typing.Dict[str, str] = {
    # Exponential
    "expon": "exp",
    # Gamma
    "gamma": "gam",
    # Generalised Extreme Value
    "genextreme": "gev",
    # Generalised Logistic
    "genlogistic": "glo",
    # Generalised Normal
    "gennorm": "gno",
    # Generalised Pareto
    "genpareto": "gpa",
    # Gumbel
    "gumbel_r": "gum",
    # Kappa
    "kappa4": "kap",
    # Normal
    "norm": "nor",
    # Pearson III
    "pearson3": "pe3",
    # Weibull
    "weibull_min": "wei",
}
_SUPPORTED_FIT_METHODS: typing.Set[str] = {"MLE", "MOM", "Lmoments"}


class Distribution:
    __slots__ = [
        "_fixed_parameters",
        "distribution",
        "distribution_parameters",
        "extremes",
        "fit_method",
        "fixed_parameters",
        "free_parameters",
        "mle_parameters",
    ]

    def __init__(
        self,
        extremes: pd.Series,
        distribution: typing.Union[str, scipy.stats.rv_continuous],
        fit_method: str = "MLE",
        **kwargs,
    ) -> None:
        """
        Distribution class compatible with pyextremes models.

        It is a wrapper around the scipy.stats.rv_continuous class and its subclasses.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html

        Parameters
        ----------
        extremes : pandas.Series
            Time series of extreme events.
        distribution : str or scipy.stats.rv_continuous
            Distribution name compatible with scipy.stats
            or a subclass of scipy.stats.rv_continuous.
            See https://docs.scipy.org/doc/scipy/reference/stats.html
        fit_method : str, default 'MLE'
            Distribution fit method.
            Controls which method is used when fitting this distribution to data.
            Supported fit methods:
             - MLE: Maximum Likelihood Estimation, from scipy
             - Lmoments: L-moments, from lmoments3
             - MOM: Method of Moments, from scipy
        kwargs
            Special keyword arguments, passed to the `.fit` method of the distribution
            (specifics depend on the `fit_method` parameter).
            These keyword arguments represent parameters to be held fixed.
            Names of parameters to be fixed must have 'f' prefixes. Valid parameters:
                - shape(s): 'fc', e.g. fc=0
                - location: 'floc', e.g. floc=0
                - scale: 'fscale', e.g. fscale=1
            By default, no parameters are fixed.
            See documentation of a specific scipy.stats distribution
            for names of available parameters.
            Lmoments does not allow fixed parameters.

        """
        self.extremes = extremes

        # Validate and set fit method
        if fit_method not in _SUPPORTED_FIT_METHODS:
            raise ValueError(
                f"Unsupported fit method '{fit_method}'. "
                f"Must be one of: {_SUPPORTED_FIT_METHODS}."
            )
        if fit_method == "Lmoments":
            if lmoments3_distr is None:  # pragma: no cover
                raise ModuleNotFoundError(
                    "The lmoments3 package is required to use method Lmoments."
                    "You may install it with `pip install pyextremes[lmoments]` or "
                    "with `pip install pyextremes[full]`."
                )
            if kwargs:
                raise ValueError("Method Lmoments does not allow fixed parameters.")
        self.fit_method = fit_method

        # Get distribution object
        self.distribution: scipy.stats.rv_continuous
        if isinstance(distribution, scipy.stats.rv_continuous):
            self.distribution = distribution
        elif isinstance(distribution, str):
            if self.fit_method == "Lmoments":
                try:
                    distribution = _SCIPY_TO_LMOMENTS3[distribution]
                except KeyError as exc:
                    raise ValueError(
                        f"Unsupported distribution for {self.fit_method}: "
                        f"'{distribution}'; "
                        f"must be one of: {', '.join(_SCIPY_TO_LMOMENTS3.keys())}."
                    ) from exc
                self.distribution = getattr(lmoments3_distr, distribution)
            else:
                self.distribution = getattr(scipy.stats, distribution)
            if not isinstance(self.distribution, scipy.stats.rv_continuous):
                raise ValueError(f"'{distribution}' is not a continuous distribution")
        else:
            raise TypeError(
                f"invalid type in {type(distribution)} for the 'distribution' argument"
            )
        self.distribution.badvalue = -np.inf
        logger.debug(
            "instantiated continuous distribution '%s'",
            self.distribution.name,
        )

        # Get a list of distribution parameter names
        self.distribution_parameters: typing.List[str] = []
        if self.distribution.shapes is not None:
            # Shape parameters must go first due to argument order in scipy.stats
            # (self.distribution_parameters is unpacked using *)
            self.distribution_parameters.extend(
                [shape.strip() for shape in self.distribution.shapes.split(",")]
            )
        self.distribution_parameters.extend(["loc", "scale"])
        valid_kwargs = [f"f{parameter}" for parameter in self.distribution_parameters]
        logger.debug(
            "collected distribution parameters: %s",
            ", ".join(self.distribution_parameters),
        )

        # Collect fixed parameters
        self.fixed_parameters: typing.Dict[str, float] = {}
        for key, value in kwargs.items():
            if key in valid_kwargs:
                self.fixed_parameters[key] = value
            else:
                raise TypeError(
                    f"'{key}' is not a valid keyword argument for "
                    f"'{self.distribution.name}' distribution, "
                    f"valid keyword arguments: {', '.join(valid_kwargs)}"
                )
        self._fixed_parameters = {
            key[1:]: value for key, value in self.fixed_parameters.items()
        }
        if len(self.fixed_parameters) == len(self.distribution_parameters):
            raise ValueError(
                "all parameters of the distribution are fixed, there is nothing to fit"
            )

        # Collect free parameters
        self.free_parameters: typing.List[str] = []
        for parameter in self.distribution_parameters:
            if parameter not in self._fixed_parameters:
                self.free_parameters.append(parameter)

        # Fit distribution
        self.mle_parameters = self.fit(data=self.extremes.values)
        free_parameters = ", ".join(
            [f"{key}={value}" for key, value in self.mle_parameters.items()]
        )
        logger.debug(
            "calculated free distribution parameters in: %s",
            free_parameters,
        )

    def fit(self, data: np.ndarray) -> typing.Dict[str, float]:
        """
        Fit distribution to data using the scipy.stats.rv_continuous.fit method.

        Parameters
        ----------
        data : numpy.ndarray
            Array with data to which the distribution is fit.

        Returns
        -------
        parameters : dict
            Dictionary with free distribution parameters.

        """

        # Calculate full distribution parameters
        if self.fit_method == "MLE":
            parameters = self.distribution.fit(
                data=data, **self.fixed_parameters, method="mle"
            )
        elif self.fit_method == "MOM":
            parameters = self.distribution.fit(
                data=data, **self.fixed_parameters, method="mm"
            )
        elif self.fit_method == "Lmoments":
            parameters = self.distribution.lmom_fit(data=data)
            parameters = list(parameters.values())
        else:
            raise RuntimeError(  # pragma: no cover
                f"Invalid fit method '{self.fit_method}'. "
                f"Must be one of: {', '.join(_SUPPORTED_FIT_METHODS)}."
            )

        # Package distribution parameters into ordered free distribution parameters
        free_parameters: typing.Dict[str, float] = {}
        for i, parameter in enumerate(self.distribution_parameters):
            if parameter in self.free_parameters:
                free_parameters[parameter] = parameters[i]
            else:
                assert np.isclose(parameters[i], self._fixed_parameters[parameter])
        return free_parameters

    @property
    def name(self) -> str:
        return self.distribution.name

    @property
    def number_of_parameters(self) -> int:
        return len(self.free_parameters)

    def __repr__(self) -> str:
        free_parameters = ", ".join(self.free_parameters)

        if len(self.fixed_parameters) == 0:
            fixed_parameters = "all parameters are free"
        else:
            fixed_parameters = ", ".join(
                [f"{key}={value:,.3f}" for key, value in self.fixed_parameters.items()]
            )

        parameters = ", ".join(
            [f"{key}={value:,.3f}" for key, value in self.mle_parameters.items()]
        )

        summary = [
            "pyextremes distribution",
            "",
            f"name: {self.name}",
            f"free parameters: {free_parameters}",
            f"fixed parameters: {fixed_parameters}",
            f"fitted parameters: {parameters}",
        ]

        longest_row = max(map(len, summary))
        summary[1] = "-" * longest_row
        summary.append(summary[1])
        summary[0] = " " * ((longest_row - len(summary[0])) // 2) + summary[0]
        for i, row in enumerate(summary):
            summary[i] += " " * (longest_row - len(row))

        return "\n".join(summary)

    def log_probability(self, theta: typing.Tuple[float, ...]) -> float:
        """
        Calculate log-probability for given free distribution parameters.

        Calculated as sum of log-prior and log-likelihood.
        Log-prior is set to 0, which corresponds to an uninformative prior.
        Log-likelihood is calculated as sum of logpdf values for a distribution
        with free parameters set to `theta` and values being extreme values.

        Parameters
        ----------
        theta : tuple
            Tuple with values of free distribution parameters.

        Returns
        -------
        logprobability : float
            log-probability for given theta.

        """
        # Unpack theta
        if len(theta) != self.number_of_parameters:
            raise ValueError(
                f"invalid theta in {theta}, "
                f"must have size of {self.number_of_parameters}"
            )
        free_parameters = dict(zip(self.free_parameters, theta))

        # Calculate log-likelihood
        return sum(
            self.distribution.logpdf(
                x=self.extremes.values, **free_parameters, **self._fixed_parameters
            )
        )

    def get_initial_state(self, n_walkers: int) -> np.ndarray:
        """
        Get initial positions for the ensemble sampler walkers.

        This method is used by the 'Emcee' model.
        Positions are sampled from a normal distribution
        for each of the free distribution parameters (e.g. c, loc, scale)
        with location for each of them taken from scipy.stats MLE fit
        and standard deviation being 0.01.

        Parameters
        ----------
        n_walkers : int
            Number of walkers used by the sampler.

        Returns
        -------
        initial_positions : numpy.ndarray
            Array with initial positions of the ensemble sampler walkers.

        """
        logger.debug("getting initial positions for %s walkers", n_walkers)
        parameters = [self.mle_parameters[key] for key in self.free_parameters]
        return scipy.stats.norm.rvs(
            loc=parameters,
            scale=0.01,
            size=(n_walkers, self.number_of_parameters),
        )

    def free2full_parameters(
        self,
        free_parameters: typing.Union[typing.Dict[str, float], np.ndarray],
    ) -> np.ndarray:
        """
        Convert container with free parameters to an array with full parameters.

        This method is used for generation of distribution parameter sequences
        to be used independently with scipy.stats.rv_continuous distributions.

        Parameters
        ----------
        free_parameters : dict or array-like
            Free parameters.
            If dict, then must be {parameter: value}.
                E.g. {'loc': 0, 'scale': 1}
            If array-like, then must either be 1D array
            or have shape of (n, number_of_free_parameters).
                E.g. [0, 1] for [loc, scale]
                or [[0, 1], [2, 3],...] for a sequence of [loc, scale] pairs
                if loc and scale are the only free parameters

        Returns
        -------
        full_parameters : numpy.ndarray
            Array with full parameters.
            Shapes:
                - 1D for dict or 1D array
                - (n, len(`.distribution_parameters`)) for array with shape
                  (n, `.number_of_parameters`)

        """
        if isinstance(free_parameters, dict):
            if len(free_parameters) != self.number_of_parameters:
                raise ValueError(
                    f"invalid value in {free_parameters} "
                    f"for the 'free_parameters' argument, "
                    f"must have length of {self.number_of_parameters}"
                )
            full_parameters = np.full(
                shape=(len(self.distribution_parameters),),
                fill_value=np.nan,
                dtype=np.float64,
            )
            for i, parameter in enumerate(self.distribution_parameters):
                try:
                    full_parameters[i] = free_parameters[parameter]
                except KeyError:
                    full_parameters[i] = self._fixed_parameters[parameter]
            return full_parameters

        else:
            # Convert 'free_parameters' argument to ndarray
            free_parameters = np.asarray(a=free_parameters, dtype=np.float64).copy()
            if free_parameters.ndim == 0:
                free_parameters = free_parameters[np.newaxis]

            if free_parameters.ndim == 1:
                # 1D array
                if len(free_parameters) != self.number_of_parameters:
                    raise ValueError(
                        f"invalid value in {free_parameters} "
                        f"for the 'free_parameters' argument, "
                        f"must have length of {self.number_of_parameters}"
                    )
                j = 0
                full_parameters = np.full(
                    shape=(len(self.distribution_parameters),),
                    fill_value=np.nan,
                    dtype=np.float64,
                )
                for i, parameter in enumerate(self.distribution_parameters):
                    try:
                        full_parameters[i] = self._fixed_parameters[parameter]
                    except KeyError:
                        full_parameters[i] = free_parameters[j]
                        j += 1
                return full_parameters

            elif free_parameters.ndim == 2:
                # 2D array
                if free_parameters.shape[1] != self.number_of_parameters:
                    raise ValueError(
                        f"invalid shape in {free_parameters.shape} "
                        f"for the 'free_parameters' argument, "
                        f"must be (n, {self.number_of_parameters})"
                    )
                j = 0
                full_parameters = np.full(
                    shape=(free_parameters.shape[0], len(self.distribution_parameters)),
                    fill_value=np.nan,
                    dtype=np.float64,
                )
                for i, parameter in enumerate(self.distribution_parameters):
                    try:
                        full_parameters[:, i] = self._fixed_parameters[parameter]
                    except KeyError:
                        full_parameters[:, i] = free_parameters[:, j]
                        j += 1
                return full_parameters

            else:
                raise ValueError(
                    f"invalid shape in {free_parameters.shape} "
                    f"for the 'free_parameters' argument, must be 1D or 2D array"
                )

    def get_prop(
        self, prop: str, x, free_parameters
    ) -> typing.Union[float, np.ndarray]:
        """
        Calculate a property such as isf, cdf, logpdf, etc.

        Parameters
        ----------
        prop : str
            Property name (e.g. 'isf' or 'logpdf').
        x : array-like
            Data for which the property is calculated.
            Scalar or 1D array-like.
        free_parameters : dict or array-like
            Dictionary or array with free distribution parameter values.
            See `.free2full_parameters` method documentation.

        Returns
        -------
        result : float or numpy.ndarray
            Calculated property value.
            If x is scalar:
                output is scalar or 1D array
                with length equal to number of `free_parameters` combinations
                    for 1D free_parameters=[1, 2] output is a scalar
                    for 2D free parameters=[[1, 2], [3, 4], [5, 6]]
                        output is an array of length len(free_parameters)
            If x is a 1D array:
                output is a 1D or 2D array
                    for 1D free_parameters=[1, 2] output is a 1D array of length len(x)
                    for 2D free_parameters=[[1, 2], [3, 4], ...]
                    output is a 2D array of shape (len(x), len(free_parameters)

        """
        # Get property function
        prop_function = getattr(self.distribution, prop)

        # Convert 'free_parameters' to an array of full parameters
        full_parameters = self.free2full_parameters(free_parameters=free_parameters)

        # Convert 'x' to ndarray
        x = np.asarray(a=x, dtype=np.float64).copy()
        if x.ndim == 0:
            x = x[np.newaxis]
        if x.ndim != 1:
            raise ValueError(
                f"invalid shape in {x.shape} for the 'x' argument, must be 1D array"
            )

        prop_values = None
        if full_parameters.ndim == 1:
            # 1D 'full_parameters' array
            prop_values = prop_function(x, *full_parameters)
        elif full_parameters.ndim == 2:
            # 2D 'full_parameters' array
            if len(x) == 1:
                full_x = x
            else:
                full_x = np.tile(x, reps=(len(full_parameters), 1))
            prop_values = prop_function(
                np.transpose(full_x), *np.transpose(full_parameters)
            )
        else:
            raise RuntimeError(
                "this is a bug: self.free2full_parameters method returned invalid value"
            )  # pragma: no cover

        if len(prop_values) == 1:
            return prop_values[0]
        else:
            return prop_values
