import calendar
import logging
import typing
import warnings

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from pyextremes.extremes import ExtremesTransformer, get_extremes, get_return_periods
from pyextremes.models import MLE, Distribution, Emcee, get_model
from pyextremes.plotting import (
    plot_corner,
    plot_extremes,
    plot_probability,
    plot_return_values,
    plot_trace,
    pyextremes_rc,
)

logger = logging.getLogger(__name__)


class EVA:

    __slots__ = [
        "__data",
        "__extremes",
        "__extremes_method",
        "__extremes_type",
        "__extremes_kwargs",
        "__extremes_transformer",
        "__model",
    ]

    def __init__(self, data: pd.Series) -> None:
        """
        Extreme Value Analysis (EVA) class.

        This class brings together most of the tools available in the pyextremes package
        bundled together in a pipeline to perform univariate extreme value analysis.

        A typical workflow using the EVA class would consist of the following:
            - extract extreme values (.get_extremes)
            - fit a model (.fit_model)
            - generate outputs (.get_summary)
            - visualize the model (.plot_diagnostic, .plot_return_values)

        Multiple additional graphical and numerical methods are available
        within this class to analyze extracted extreme values, visualize them,
        assess goodness-of-fit of selected model, and to visualize its outputs.

        Parameters
        ----------
        data : pandas.Series
            Time series to be analyzed.
            Index must be date-time and values must be numeric.

        """
        # Ensure that 'data' is pandas Series
        if not isinstance(data, pd.Series):
            raise TypeError(
                f"invalid type in {type(data)} for the 'data' argument, "
                f"must be pandas.Series"
            )

        # Copy 'data' to ensure the original Series object it is not mutated
        data = data.copy(deep=True)

        # Ensure that 'data' has correct index and value dtypes
        if not np.issubdtype(data.dtype, np.number):
            try:
                data = data.astype(np.float64)
                message = "'data' values are not numeric - converted to numeric"
                logger.info(message)
                warnings.warn(message=message, category=RuntimeWarning)
            except ValueError as _error:
                raise TypeError(
                    f"invalid dtype in {data.dtype} for the 'data' argument, "
                    f"must be numeric (subdtype of numpy.number)"
                ) from _error
        if not data.index.is_all_dates:
            raise TypeError(
                f"index of 'data' must be a sequence of date-time objects, "
                f"not {data.index.dtype}"
            )

        # Ensure that 'data' is sorted
        if not data.index.is_monotonic_increasing:
            message = (
                "'data' index is not sorted in ascending order - sorting data by index"
            )
            logger.info(message)
            warnings.warn(message=message, category=RuntimeWarning)
            data = data.sort_index(ascending=True)

        # Ensure that 'data' has no invalid entries
        n_nans = data.isna().sum()
        if n_nans > 0:
            message = (
                f"{n_nans:,d} Null values found in 'data' - removing invalid entries"
            )
            logger.info(message)
            warnings.warn(message=message, category=RuntimeWarning)
            data = data.dropna()

        # Set the 'data' attribute
        self.__data = data

        # Initialize attributes related to extreme value extraction
        self.__extremes: typing.Optional[pd.Series] = None
        self.__extremes_method: typing.Optional[str] = None
        self.__extremes_type: typing.Optional[str] = None
        self.__extremes_kwargs: typing.Optional[dict] = None
        self.__extremes_transformer: typing.Optional[ExtremesTransformer] = None

        # Initialize attributes related to model fitting
        self.__model: typing.Optional[typing.Union[MLE, Emcee]] = None

        logger.info(
            f"successfully initialized EVA object "
            f"with data of length {len(self.data):,d}"
        )

    @property
    def data(self) -> pd.Series:
        return self.__data

    def __get_extremes_attribute(self, attribute_name: str) -> typing.Any:
        value = getattr(self, f"_EVA__{attribute_name}")
        if value is None:
            raise AttributeError(
                "extreme values must first be extracted "
                "using the '.get_extremes' method"
            )
        else:
            return value

    @property
    def extremes(self) -> pd.Series:
        return self.__get_extremes_attribute("extremes")

    @property
    def extremes_method(self) -> str:
        return self.__get_extremes_attribute("extremes_method")

    @property
    def extremes_type(self) -> str:
        return self.__get_extremes_attribute("extremes_type")

    @property
    def extremes_kwargs(self) -> typing.Dict[str, typing.Any]:
        return self.__get_extremes_attribute("extremes_kwargs")

    @property
    def extremes_transformer(self) -> ExtremesTransformer:
        return self.__get_extremes_attribute("extremes_transformer")

    @property
    def model(self) -> typing.Union[MLE, Emcee]:
        if self.__model is None:
            raise AttributeError(
                "model must first be assigned using the '.fit_model' method"
            )
        else:
            return self.__model

    @property
    def distribution(self) -> Distribution:
        return self.model.distribution

    @property
    def loglikelihood(self) -> float:
        return self.model.loglikelihood

    @property
    def AIC(self) -> float:
        return self.model.AIC

    # TODO
    def __repr__(self) -> str:
        """Representation of the class state."""
        # Width of repr block
        width = 101

        # Separator used to separate two columns of the repr block
        sep = " " * 6

        # Widths of left and ridght columns
        lwidth = (width - len(sep)) // 2
        rwidth = width - (lwidth + len(sep))

        # Function used to center text within repr block line
        def center_text(text: str) -> str:
            left_gap = (width - len(text)) // 2
            right_gap = width - left_gap - len(text)
            return "".join([" " * left_gap, text, " " * right_gap])

        # Function used to convert label-value pair into a sequence of lines within a column
        def align_text(text: str, value: str, position: str = "left") -> str:
            if text == "":
                if position == "left":
                    return f"{value:>{lwidth:d}}"
                elif position == "right":
                    return f"{value:>{rwidth:d}}"
                else:
                    raise ValueError

            text_width = len(text) + 2
            if position == "left":
                free_width = lwidth - text_width
            else:
                free_width = rwidth - text_width

            value_chunks = [
                value[i : i + free_width] for i in range(0, len(value), free_width)
            ]
            if "".join(value_chunks) != value:
                raise AssertionError

            aligned_text = []
            for i, chunk in enumerate(value_chunks):
                if i == 0:
                    aligned_text.append(f"{text}: {chunk:>{free_width:d}}")
                else:
                    aligned_text.append(
                        "".join([" " * text_width, f"{chunk:>{free_width:d}}"])
                    )
            return "\n".join(aligned_text)

        # Function used to convert a pair pair of label-value pairs
        # into a sequence of lines of the repr block
        def align_pair(text: tuple, value: tuple) -> str:
            left_part = align_text(text=text[0], value=value[0], position="left").split(
                "\n"
            )
            right_part = align_text(
                text=text[1], value=value[1], position="right"
            ).split("\n")

            if len(left_part) < len(right_part):
                for _ in range(len(right_part) - len(left_part)):
                    left_part.append(" " * len(left_part[0]))
            else:
                for _ in range(len(left_part) - len(right_part)):
                    right_part.append(" " * len(right_part[0]))

            aligned_text = []
            for left, right in zip(left_part, right_part):
                aligned_text.append("".join([left, sep, right]))

            return "\n".join(aligned_text)

        # summary header
        start_date = (
            f"{calendar.month_name[self.data.index[0].month]} {self.data.index[0].year}"
        )
        end_date = f"{calendar.month_name[self.data.index[-1].month]} {self.data.index[-1].year}"
        summary = [
            center_text("Univariate Extreme Value Analysis"),
            "=" * width,
            center_text("Original Data"),
            "-" * width,
            align_pair(
                ("Data label", "Data range"),
                (str(self.data.name), f"{start_date} to {end_date}"),
            ),
            "=" * width,
            center_text("Extreme Values"),
            "-" * width,
        ]

        # extremes section
        if self.extremes is None:
            summary.extend(["Extreme values have not been extracted", "=" * width])
        else:
            if self.extremes_method == "BM":
                ev_parameters = ("Block size", str(self.extremes_kwargs["block_size"]))
            elif self.extremes_method == "POT":
                ev_parameters = ("Threshold", str(self.extremes_kwargs["threshold"]))
            else:
                raise RuntimeError("this is a bug - wrong extremes method")
            summary.extend(
                [
                    align_pair(
                        ("Number of extreme events", "Extraction method"),
                        (f"{len(self.extremes):d}", str(self.extremes_method)),
                    ),
                    align_pair(
                        ("Type of extreme events", ev_parameters[0]),
                        (str(self.extremes_type), ev_parameters[1]),
                    ),
                    "=" * width,
                ]
            )

        # model section
        summary.extend(
            [
                center_text("Model"),
                "-" * width,
            ]
        )
        if self.model is None:
            summary.extend(["Model has not been fit to the extremes", "=" * width])
        else:
            summary.append(
                align_pair(
                    ("Model", "Distribution"),
                    (self.model.name, self.model.distribution.name),
                )
            )
            if self.model.name == "Emcee":
                summary.append(
                    align_pair(
                        ("Walkers", "Samples per walker"),
                        (f"{self.model.n_walkers:d}", f"{self.model.n_samples:d}"),
                    )
                )

            summary.append(
                align_pair(
                    ("Log-likelihood", "AIC"),
                    (f"{self.model.loglikelihood:.3f}", f"{self.model.AIC:.3f}"),
                )
            )

            summary.append("-" * width)

            free_parameters = [
                f"{parameter}={self.model.fit_parameters[parameter]:.3f}"
                for parameter in self.model.distribution.free_parameters
            ]
            fixed_parameters = [
                f"{key}={value:.3f}"
                for key, value in self.model.distribution.fixed_parameters.items()
            ]
            if len(fixed_parameters) == 0:
                fixed_parameters = ["All parameters are free"]
            if len(free_parameters) < len(fixed_parameters):
                for _ in range(len(fixed_parameters) - len(free_parameters)):
                    free_parameters.append("")
            elif len(fixed_parameters) < len(free_parameters):
                for _ in range(len(free_parameters) - len(fixed_parameters)):
                    fixed_parameters.append("")
            else:
                pass
            for j, (frp, fip) in enumerate(zip(free_parameters, fixed_parameters)):
                if j == 0:
                    summary.append(
                        align_pair(
                            text=("Free parameters", "Fixed parameters"),
                            value=(frp, fip),
                        )
                    )
                else:
                    summary.append(align_pair(text=("", ""), value=(frp, fip)))
            summary.append("=" * width)
        return "\n".join(summary)

    def get_extremes(self, method: str, extremes_type: str = "high", **kwargs) -> None:
        """
        Get extreme events from time series.

        Extracts extreme values from the 'self.data' attribute.
        Stores extreme values in the 'self.extremes' attribute.

        Parameters
        ----------
        method : str
            Extreme value extraction method.
            Supported values:
                BM - Block Maxima
                POT - Peaks Over Threshold
        extremes_type : str, optional
            high (default) - get extreme high values
            low - get extreme low values
        kwargs
            if method is BM:
                block_size : str or pandas.Timedelta, optional
                    Block size (default='1Y').
                errors : str, optional
                    raise (default) - raise an exception
                        when encountering a block with no data
                    ignore - ignore blocks with no data
                    coerce - get extreme values for blocks with no data
                        as mean of all other extreme events in the series
                        with index being the middle point of corresponding interval
            if method is POT:
                threshold : float
                    Threshold used to find exceedances.
                r : str or pandas.Timedelta, optional
                    Duration of window used to decluster the exceedances.
                    By default r='24H' (24 hours).

        """
        message = f"for method='{method}' and extremes_type='{extremes_type}'"
        logger.debug(f"extracting extreme values {message}")
        self.__extremes = get_extremes(
            method=method, ts=self.data, extremes_type=extremes_type, **kwargs
        )
        self.__extremes_method = method
        self.__extremes_type = extremes_type
        logger.info(f"successfully extracted extreme values {message}")

        logger.debug("collecting extreme value properties ")
        self.__extremes_kwargs = {}
        if method == "BM":
            self.__extremes_kwargs["block_size"] = pd.to_timedelta(
                kwargs.get("block_size", "1Y")
            )
            self.__extremes_kwargs["errors"] = kwargs.get("errors", "raise")
        elif method == "POT":
            self.__extremes_kwargs["threshold"] = kwargs["threshold"]
            self.__extremes_kwargs["r"] = pd.to_timedelta(kwargs.get("r", "24H"))
        else:
            raise AssertionError
        logger.info("successfully collected extreme value properties")

        logger.debug("creating extremes transformer")
        self.__extremes_transformer = ExtremesTransformer(
            extremes=self.__extremes,
            extremes_type=self.__extremes_type,
        )
        logger.info("successfully created extremes transformer")

        logger.info("removing any previously declared models")
        self.__model = None

    def plot_extremes(
        self,
        figsize: tuple = (8, 5),
        ax: typing.Optional[plt.Axes] = None,
    ) -> typing.Tuple[plt.Figure, plt.Axes]:  # pragma: no cover
        """
        Plot extreme events.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size in inches in format (width, height).
            By default it is (8, 5).
        ax : matplotlib.axes._axes.Axes, optional
            Axes onto which extremes plot is drawn.
            If None (default), a new figure and axes objects are created.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : matplotlib.axes._axes.Axes
            Axes object.

        """
        return plot_extremes(
            ts=self.data,
            extremes=self.extremes,
            extremes_method=self.extremes_method,
            extremes_type=self.extremes_type,
            block_size=self.extremes_kwargs.get("block_size", None),
            threshold=self.extremes_kwargs.get("threshold", None),
            figsize=figsize,
            ax=ax,
        )

    def fit_model(
        self,
        model: str = "MLE",
        distribution: typing.Union[str, scipy.stats.rv_continuous] = None,
        distribution_kwargs: typing.Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Fit a model to the extracted extreme values.

        Parameters
        ----------
        model : str, optional
            Name of model. By default it is 'MLE'.
            Name of model.
            Supported models:
                MLE - Maximum Likelihood Estimate (MLE) model.
                    Based on 'scipy' package (scipy.stats.rv_continuous.fit).
                Emcee - Markov Chain Monte Carlo (MCMC) model.
                    Based on 'emcee' package by Daniel Foreman-Mackey.
        distribution : str or scipy.stats.rv_continuous, optional
            Distribution name compatible with scipy.stats
            or a subclass of scipy.stats.rv_continuous.
            See https://docs.scipy.org/doc/scipy/reference/stats.html
            By default the distribution is selected automatically
            as best between 'genextreme' and 'gumbel_r' for 'BM' extremes
            and 'genpareto' and 'expon' for 'POT' extremes.
            Best distribution is selected using the AIC metric.
        distribution_kwargs : dict, optional
            Special keyword arguments, passsed to the `.fit` method of the distribution.
            These keyword arguments represent parameters to be held fixed.
            Names of parameters to be fixed must have 'f' prefixes. Valid parameters:
                - shape(s): 'fc', e.g. fc=0
                - location: 'floc', e.g. floc=0
                - scale: 'fscale', e.g. fscale=1
            See documentation of a specific scipy.stats distribution
            for names of available parameters.
            By default, location parameter for 'genpareto' and 'expon' distributions
            is fixed to threshold (POT) or to minimum extremes (BM) value.
            Set to empty dictionary (distribution_kwargs={}) to avoid this behaviour.
        kwargs
            Keyword arguments passed to a model .fit method.
            MLE model:
                MLE model takes no additional arguments.
            Emcee model:
                n_walkers : int, optional
                    The number of walkers in the ensemble (default=100).
                n_samples : int, optional
                    The number of steps to run (default=500).
                progress : bool or str, optional
                    If True, a progress bar will be shown as the sampler progresses.
                    If a string, will select a specific tqdm progress bar.
                    Most notable is 'notebook', which shows a progress bar
                    suitable for Jupyter notebooks.
                    If False (default), no progress bar will be shown.
                    This progress bar is a part of the `emcee` package.

        """
        # Select default distribution
        if distribution is None:
            logger.debug(
                f"selecting default distribution for extremes extracted using the "
                f"'{self.extremes_method}' method"
            )

            # Prepare list of candidate distributions
            if self.extremes_method == "BM":
                candidate_distributions = ["genextreme", "gumbel_r"]
                _distribution_kwargs = None
            elif self.extremes_method == "POT":
                candidate_distributions = ["genpareto", "expon"]
                _distribution_kwargs = {
                    "floc": self.extremes_kwargs.get(
                        "threshold",
                        self.extremes_transformer.transformed_extremes.min(),
                    )
                }
            else:
                raise AssertionError

            # Fit MLE model for candidate distributions
            # and select distribution with smallest AIC
            candidate_models = {
                distribution_name: MLE(
                    extremes=self.extremes_transformer.transformed_extremes,
                    distribution=distribution_name,
                    distribution_kwargs=_distribution_kwargs,
                ).AIC
                for distribution_name in candidate_distributions
            }
            distribution = min(candidate_models, key=candidate_models.get)
            logger.info(
                f"selected '{distribution}' distribution "
                f"with AIC score {candidate_models[distribution]}"
            )

        # Get distribution name
        if isinstance(distribution, str):
            distribution_name = distribution
        elif isinstance(distribution, scipy.stats.rv_continuous):
            distribution_name = getattr(distribution, "name", None)
        else:
            raise TypeError(
                f"invalid type in {type(distribution)} "
                f"for the 'distribution' argument, "
                f"must be string or scipy.stats.rv_continuous"
            )

        # Checking if distribution is valid per extreme value theory:
        # Fisher-Tippet-Gnedenko theorem for 'BM'
        # Pickands–Balkema–de Haan theorem for 'POT'
        if distribution_name is None:
            warnings.warn(
                message=(
                    "provided distribution 'name' attribute cannot be resolved "
                    "and distribution validity cannot be verified"
                ),
                category=RuntimeWarning,
            )
        else:
            if self.extremes_method == "BM" and distribution_name not in [
                "genextreme",
                "gumbel_r",
            ]:
                warnings.warn(
                    message=(
                        f"'{distribution_name}' distribution is not "
                        f"recommended to be used with extremes extracted "
                        f"using the 'BM' method, 'genextreme' or 'gumebel_r' "
                        f"should be used per the Fisher-Tippet-Gnedenko theorem"
                    ),
                    category=RuntimeWarning,
                )
            elif self.extremes_method == "POT" and distribution_name not in [
                "genpareto",
                "expon",
            ]:
                warnings.warn(
                    message=(
                        f"'{distribution_name}' distribution is not "
                        f"recommended to be used with extremes extracted "
                        f"using the 'POT' method, 'genpareto' or 'expon' "
                        f"should be used per the Pickands–Balkema–de Haan theorem"
                    ),
                    category=RuntimeWarning,
                )

        # Freeze (fix) location parameter for genpareto/expon distributions
        if distribution_kwargs is None and distribution_name in ["genpareto", "expon"]:
            distribution_kwargs = {
                "floc": self.extremes_kwargs.get(
                    "threshold", self.extremes_transformer.transformed_extremes.min()
                )
            }
            logger.debug(
                f"freezing location parameter (floc) at {distribution_kwargs['floc']} "
                f"for '{distribution_name}' distribution"
            )

        # Fit model to transformed extremes
        self.__model = get_model(
            model=model,
            extremes=self.extremes_transformer.transformed_extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            **kwargs,
        )

    def _get_mcmc_plot_inputs(self, labels=None) -> tuple:
        try:
            trace = self.model.trace
            trace_map = tuple(
                self.model.fit_parameters[parameter]
                for parameter in self.model.distribution.free_parameters
            )
        except TypeError as _error:
            raise TypeError(
                f"this method is only applicable to MCMC-like models, "
                f"not to '{self.model.name}' model"
            ) from _error

        parameter_names = {
            "loc": r"Location, $\mu$",
            "scale": r"Scale, $\sigma$",
        }
        if self.model.distribution.name in ["genextreme", "genpareto"]:
            parameter_names["c"] = r"Shape, $\xi$"
        if labels is None:
            labels = []
            for parameter in self.model.distribution.free_parameters:
                try:
                    labels.append(parameter_names[parameter])
                except KeyError:
                    labels.append(f"Shape parameter '{parameter}'")

        return trace, trace_map, labels

    def plot_trace(
        self,
        burn_in: int = 0,
        labels=None,
        figsize: typing.Optional[tuple] = None,
    ) -> typing.Tuple[plt.Figure, list]:  # pragma: no cover
        """
        Plot trace plot for MCMC sampler trace.

        Parameters
        ----------
        burn_in : int, optional
            Burn-in value (number of first steps to discard for each walker).
            By default it is 0 (no values are discarded).
        labels : array-like, optional
            Sequence of strings with parameter names, used to label axes.
            If None (default), then axes are labeled sequentially.
        figsize : tuple, optional
            Figure size in inches.
            If None (default), then figure size is calculated automatically
            as 8 by 2 times number of parameters.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : list
            List with n_parameters Axes objects.

        """
        trace, trace_map, labels = self._get_mcmc_plot_inputs(labels=labels)
        return plot_trace(
            trace=trace,
            trace_map=trace_map,
            burn_in=burn_in,
            labels=labels,
            figsize=figsize,
        )

    def plot_corner(
        self,
        burn_in: int = 0,
        labels=None,
        levels: typing.Optional[int] = None,
        figsize: tuple = (8, 8),
    ) -> typing.Tuple[plt.Figure, list]:  # pragma: no cover
        """
        Plot corner plot for MCMC sampler trace.

        Parameters
        ----------
        burn_in : int, optional
            Burn-in value (number of first steps to discard for each walker).
            By default it is 0 (no values are discarded).
        labels : array-like, optional
            Sequence of strings with parameter names, used to label axes.
            If None (default), then axes are labeled sequentially.
        levels : int, optional
            Number of Gaussian KDE contours to plot.
            If None (default), then not shown.
        figsize : tuple, optional
            Figure size in inches. By default it is (8, 8).

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : list
            2D list with Axes objects
            of size N by N, where N is n_parameters.

        """
        trace, trace_map, labels = self._get_mcmc_plot_inputs(labels=labels)
        return plot_corner(
            trace=trace,
            trace_map=trace_map,
            burn_in=burn_in,
            labels=labels,
            levels=levels,
            figsize=figsize,
        )

    def get_return_value(
        self,
        return_period: typing.Union[float, tuple, list, np.ndarray],
        return_period_size: typing.Union[str, pd.Timedelta] = "1Y",
        alpha: float = None,
        **kwargs,
    ) -> tuple:
        """
        Get return value and confidence interval for given return period(s).

        Parameters
        ----------
        return_period : float or array-like
            Return period or array of return periods.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=None).
            If None, return None for upper and lower confidence interval bounds.
        kwargs
            Keyword arguments specific to a model.
            If alpha is None, keyword arguments are ignored (error still raised for unrecognized arguments).
            MLE model:
                n_samples : int, optional
                    Number of bootstrap samples used to estimate confidence interval bounds (default=100).
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        return_value : float or array-like
            Return value(s).
        lower_ci_bound : float or array-like
            Lower confidence interval bound(s).
        upper_ci_bount : float or array-like
            Upper confidence interval bound(s).
        """
        logger.info("making sure a model has been fit")
        if self.model is None:
            raise AttributeError(
                "a model must be fit to extracted extremes first, use .fit_model method"
            )

        logger.info(
            "calculating rate of extreme events as number of events per return_period_size"
        )
        if self.extremes_method == "BM":
            extremes_rate = return_period_size / self.extremes_kwargs["block_size"]
            logger.debug("calculated extremes_rate for BM method")
        elif self.extremes_method == "POT":
            n_periods = (self.data.index[-1] - self.data.index[0]) / return_period_size
            extremes_rate = len(self.extremes) / n_periods
            logger.debug("calculated extremes_rate for POT method")
        else:
            raise RuntimeError("this is a bug - invalid extremes method")

        logger.info("calculating exceedance probability")
        if isinstance(return_period, (tuple, list, np.ndarray)):
            logger.info("getting a list of exceedance probabilities")
            exceedance_probability = 1 / np.array(return_period) / extremes_rate
        elif isinstance(return_period, (int, float, np.number)):
            logger.info("getting a single exceedance probability")
            exceedance_probability = 1 / return_period / extremes_rate
        else:
            raise TypeError(
                f"invalid type in {type(return_period)} for the 'return_period' argument"
            )

        logger.info("calculating return value using the model")
        rv = self.model.get_return_value(
            exceedance_probability=exceedance_probability, alpha=alpha, **kwargs
        )
        return tuple(self.extremes_transformer.transform(value) for value in rv)

    def get_summary(
        self,
        return_period: typing.Union[tuple, list, np.ndarray],
        return_period_size: typing.Union[str, pd.Timedelta] = "1Y",
        alpha: float = 0.95,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with return values and confidence interval bounds for given return periods.

        Parameters
        ----------
        return_period : array-like
            Return period or array of return periods.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=0.95).
        kwargs
            Model-specific keyword arguments.
            MLE model:
                n_samples : int
                    Number of samles used to get confidence interval.
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        summary : pandas.DataFrame
            DataFrame with return values and confidence interval bounds.
        """
        logger.info("calculating return values")
        return_values = self.get_return_value(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=alpha,
            **kwargs,
        )

        logger.info("preparing the summary dataframe")
        return pd.DataFrame(
            data=np.transpose(return_values),
            index=pd.Index(data=return_period, name="return period"),
            columns=["return value", "lower ci", "upper ci"],
        )

    def plot_return_values(
        self,
        return_period: typing.Iterable = None,
        return_period_size: typing.Union[str, pd.Timedelta] = "1Y",
        alpha: float = 0.95,
        plotting_position: str = "weibull",
        ax=None,
        figsize: tuple = (8, 8 / 1.618),
        **kwargs,
    ) -> tuple:  # pragma: no cover
        """
        Plot return values and confidence intervals for given return periods.

        Parameters
        ----------
        return_period : array-like, optional
            Return period or array of return periods.
            If None, calculates as 100 values uniformly spaced within the range
            of return periods of the extracted extreme values.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=0.95).
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        ax : matplotlib.axes.Axes, optional
            Axes onto which the figure is drawn (default=None).
            If None, a new figure and axes is created.
        figsize : tuple, optional
            Figure size in inches (default=(8, 8/1.618)).
        kwargs
            Model-specific keyword arguments.
            MLE model:
                n_samples : int
                    Number of samles used to get confidence interval.
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        if ax is None:
            figure : matplotlib.figure.Figure
                Figure object.
        else:
            None
        axes : matplotlib.axes.Axes
            Axes object.
        """
        logger.info("getting observed return values")
        try:
            block_size = self.extremes_kwargs["block_size"]
        except KeyError:
            block_size = None
        observed_return_values = get_return_periods(
            ts=self.data,
            extremes=self.extremes,
            extremes_method=self.extremes_method,
            extremes_type=self.extremes_type,
            block_size=block_size,
            return_period_size=return_period_size,
            plotting_position=plotting_position,
        )

        if return_period is None:
            logger.info("creating default return_period array")
            return_period = np.linspace(
                observed_return_values.loc[:, "return period"].min(),
                observed_return_values.loc[:, "return period"].max(),
                100,
            )

        logger.info("getting modeled return values")
        modeled_return_values = self.get_summary(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=alpha,
            **kwargs,
        )

        logger.info("plotting return values")
        return plot_return_values(
            observed_return_values=observed_return_values,
            modeled_return_values=modeled_return_values,
            ax=ax,
            figsize=figsize,
        )

    def plot_probability(
        self,
        plot_type: str,
        return_period_size: typing.Union[str, pd.Timedelta] = "1Y",
        plotting_position: str = "weibull",
        ax=None,
        figsize: tuple = (8, 8),
    ) -> tuple:  # pragma: no cover
        """
        Plot a probability plot (QQ or PP).

        Used to assess goodness-of-fit of the model.

        Parameters
        ----------
        plot_type : str
            Probability plot type. Valid values are PP and QQ.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        ax : matplotlib.axes.Axes, optional
            Axes onto which the figure is drawn (default=None).
            If None, a new figure and axes is created.
        figsize : tuple, optional
            Figure size in inches (default=(8, 8)).

        Returns
        -------
        if ax is None:
            figure : matplotlib.figure.Figure
                Figure object.
        else:
            None
        axes : matplotlib.axes.Axes
            Axes object.
        """
        logger.info("getting observed return values")
        try:
            block_size = self.extremes_kwargs["block_size"]
        except KeyError:
            block_size = None
        observed_return_values = get_return_periods(
            ts=self.data,
            extremes=self.extremes,
            extremes_method=self.extremes_method,
            extremes_type=self.extremes_type,
            block_size=block_size,
            return_period_size=return_period_size,
            plotting_position=plotting_position,
        )

        logger.info(
            f"getting observed and theoretical values for plot_type='{plot_type}'"
        )
        if plot_type == "PP":
            observed = (
                1 - observed_return_values.loc[:, "exceedance probability"].values
            )
            theoretical = self.model.cdf(
                self.extremes_transformer.transform(
                    observed_return_values.loc[:, self.extremes.name].values
                )
            )
        elif plot_type == "QQ":
            observed = observed_return_values.loc[:, self.extremes.name].values
            theoretical = self.extremes_transformer.transform(
                self.model.isf(
                    observed_return_values.loc[:, "exceedance probability"].values
                )
            )
        else:
            raise ValueError(
                f"'{plot_type}' is not a valid 'plot_type' value. Available plot_types: PP, QQ"
            )

        logger.info("plotting the probability plot")
        return plot_probability(
            observed=observed, theoretical=theoretical, ax=ax, figsize=figsize
        )

    def plot_diagnostic(
        self,
        return_period: typing.Iterable = None,
        return_period_size: typing.Union[str, pd.Timedelta] = "1Y",
        alpha: float = 0.95,
        plotting_position: str = "weibull",
        figsize: tuple = (8, 8),
        **kwargs,
    ):  # pragma: no cover
        """
        Plot a diagnostic plot.

        This plot can be used to quickly assess goodness-of-fit of the selected model.
        The diagnostice plot consists of four axes:
            - top left : return values
            - top right : PDF
            - bottom left : Q-Q
            - bottom right : P-P

        Parameters
        ----------
        return_period : array-like, optional
            Return period or array of return periods.
            If None, calculates as 100 values uniformly spaced within the range
            of return periods of the extracted extreme values.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='1Y').
            If set to '30D', then a return period of 12 would be equivalent to 1 year return period.
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        alpha : float, optional
            Width of confidence interval, from 0 to 1 (default=0.95).
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        figsize : tuple, optional
            Figure size in inches (default=(8, 8)).
        kwargs
            Model-specific keyword arguments.
            MLE model:
                n_samples : int
                    Number of samles used to get confidence interval.
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : tuple
            Tuple with four Axes objects: return values, pdf, qq, pp
        """
        with plt.rc_context(rc=pyextremes_rc):
            logger.info("creating figure")
            fig = plt.figure(figsize=figsize, dpi=96)

            logger.info("creating gridspec")
            gs = matplotlib.gridspec.GridSpec(
                nrows=2,
                ncols=2,
                wspace=0.3,
                hspace=0.3,
                width_ratios=[1, 1],
                height_ratios=[1, 1],
            )

            logger.info("creating axes")
            ax_rv = fig.add_subplot(gs[0, 0])
            ax_pdf = fig.add_subplot(gs[0, 1])
            ax_qq = fig.add_subplot(gs[1, 0])
            ax_pp = fig.add_subplot(gs[1, 1])

            logger.info("plotting return values")
            self.plot_return_values(
                return_period=return_period,
                return_period_size=return_period_size,
                alpha=alpha,
                plotting_position=plotting_position,
                ax=ax_rv,
                **kwargs,
            )
            ax_rv.set_title("Return value plot")
            ax_rv.grid(False, which="both")

            logger.info("plotting pdf")
            pdf_support = np.linspace(self.extremes.min(), self.extremes.max(), 100)
            pdf = self.model.pdf(self.extremes_transformer.transform(pdf_support))
            ax_pdf.grid(False)
            ax_pdf.set_title("Probability density plot")
            ax_pdf.set_ylabel("Probability density")
            ax_pdf.set_xlabel(self.data.name)
            ax_pdf.hist(
                self.extremes.values,
                bins=np.histogram_bin_edges(a=self.extremes.values, bins="auto"),
                density=True,
                rwidth=0.8,
                facecolor="#5199FF",
                edgecolor="None",
                lw=0,
                alpha=0.25,
                zorder=5,
            )
            ax_pdf.hist(
                self.extremes.values,
                bins=np.histogram_bin_edges(a=self.extremes.values, bins="auto"),
                density=True,
                rwidth=0.8,
                facecolor="None",
                edgecolor="#5199FF",
                lw=1,
                ls="--",
                zorder=10,
            )
            ax_pdf.plot(pdf_support, pdf, color="#F85C50", lw=2, ls="-", zorder=15)
            ax_pdf.scatter(
                self.extremes.values,
                np.full(shape=len(self.extremes), fill_value=0),
                marker="|",
                s=40,
                facecolor="k",
                edgecolor="None",
                lw=0.5,
                zorder=15,
            )
            ax_pdf.set_ylim(0, ax_pdf.get_ylim()[1])

            logger.info("plotting Q-Q plot")
            self.plot_probability(
                plot_type="QQ",
                return_period_size=return_period_size,
                plotting_position=plotting_position,
                ax=ax_qq,
            )
            ax_qq.set_title("Q-Q plot")

            logger.info("plotting P-P plot")
            self.plot_probability(
                plot_type="PP",
                return_period_size=return_period_size,
                plotting_position=plotting_position,
                ax=ax_pp,
            )
            ax_pp.set_title("P-P plot")

            return fig, (ax_rv, ax_pdf, ax_qq, ax_pp)
