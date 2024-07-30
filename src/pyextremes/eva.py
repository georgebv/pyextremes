from __future__ import annotations

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
from pyextremes.tests import KolmogorovSmirnov

logger = logging.getLogger(__name__)


class EVA:
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
    """

    __slots__ = [
        "__data",
        "__extremes",
        "__extremes_method",
        "__extremes_type",
        "__extremes_kwargs",
        "__extremes_transformer",
        "__model",
    ]

    __data: pd.Series
    __extremes: typing.Optional[pd.Series]
    __extremes_method: typing.Optional[typing.Literal["BM", "POT"]]
    __extremes_type: typing.Optional[typing.Literal["high", "low"]]
    __extremes_kwargs: typing.Optional[typing.Dict[str, typing.Any]]
    __extremes_transformer: typing.Optional[ExtremesTransformer]
    __model: typing.Optional[typing.Union[MLE, Emcee]]

    def __init__(self, data: pd.Series) -> None:
        """
        Initialize EVA model.

        Parameters
        ----------
        data : pandas.Series
            Time series to be analyzed.
            Index must be date-time and values must be numeric.

        """
        # Ensure that `data` is pandas Series
        if not isinstance(data, pd.Series):
            raise TypeError(
                f"invalid type in '{type(data).__name__}' for the `data` argument, "
                f"must be pandas.Series"
            )

        # Copy `data` to ensure the original Series object it is not mutated
        data = data.copy(deep=True)

        # Ensure that `data` has correct index and value dtypes
        if not np.issubdtype(data.dtype, np.number):
            try:
                message = "`data` values are not numeric - converting to numeric"
                logger.debug(message)
                warnings.warn(message=message, category=RuntimeWarning)
                data = data.astype(np.float64)
            except ValueError as _error:
                raise TypeError(
                    f"invalid dtype in {data.dtype} for the `data` argument, "
                    f"must be numeric (subdtype of numpy.number)"
                ) from _error
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(
                f"index of `data` must be a sequence of date-time objects, "
                f"not {data.index.inferred_type}"
            )

        # Ensure `data` doesn't have duplicate indices
        if (n_duplicates := len(data) - len(data.index.drop_duplicates())) > 0:
            message = (
                f"{n_duplicates:,d} duplicate indices found in `data` "
                "- removing duplicate entries"
            )
            logger.debug(message)
            warnings.warn(message=message, category=RuntimeWarning)
            data = data.groupby(data.index).first()

        # Ensure that `data` is sorted
        if not data.index.is_monotonic_increasing:
            message = (
                "`data` index is not sorted in ascending order - "
                "sorting `data` by index"
            )
            logger.debug(message)
            warnings.warn(message=message, category=RuntimeWarning)
            data = data.sort_index(ascending=True)

        # Ensure that `data` has no invalid entries
        n_nans = data.isna().sum()
        if n_nans > 0:
            message = (
                f"{n_nans:,d} Null values found in `data` - removing invalid entries"
            )
            logger.debug(message)
            warnings.warn(message=message, category=RuntimeWarning)
            data = data.dropna()

        # Set the `data` attribute
        self.__data: pd.Series = data

        # Initialize attributes related to extreme value extraction
        self.__extremes = None
        self.__extremes_method = None
        self.__extremes_type = None
        self.__extremes_kwargs = None
        self.__extremes_transformer = None

        # Initialize attributes related to model fitting
        self.__model = None

        logger.info("successfully initialized EVA object")

    @property
    def data(self) -> pd.Series:
        return self.__data

    @property
    def extremes(self) -> pd.Series:
        if self.__extremes is None:
            raise AttributeError(
                "extreme values must first be extracted "
                "using the '.get_extremes' method"
            )
        return self.__extremes

    @property
    def extremes_method(self) -> typing.Literal["BM", "POT"]:
        if self.__extremes_method is None:
            raise AttributeError(
                "extreme values must first be extracted "
                "using the '.get_extremes' method"
            )
        return self.__extremes_method

    @property
    def extremes_type(self) -> typing.Literal["high", "low"]:
        if self.__extremes_type is None:
            raise AttributeError(
                "extreme values must first be extracted "
                "using the '.get_extremes' method"
            )
        return self.__extremes_type

    @property
    def extremes_kwargs(self) -> typing.Dict[str, typing.Any]:
        if self.__extremes_kwargs is None:
            raise AttributeError(
                "extreme values must first be extracted "
                "using the '.get_extremes' method"
            )
        return self.__extremes_kwargs

    @property
    def extremes_transformer(self) -> ExtremesTransformer:
        if self.__extremes_transformer is None:
            raise AttributeError(
                "extreme values must first be extracted "
                "using the '.get_extremes' method"
            )
        return self.__extremes_transformer

    @property
    def model(self) -> typing.Union[MLE, Emcee]:
        if self.__model is None:
            raise AttributeError(
                "model must first be assigned using the '.fit_model' method"
            )
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

    def test_ks(self, significance_level: float = 0.05) -> KolmogorovSmirnov:
        return KolmogorovSmirnov(
            extremes=self.extremes_transformer.transformed_extremes,
            distribution=self.distribution.distribution,
            fit_parameters={
                **self.model.fit_parameters,
                **self.model.distribution._fixed_parameters,
            },
            significance_level=significance_level,
        )

    def __repr__(self) -> str:
        # Width of repr block
        width = 88

        # Separator used to separate two columns of the repr block
        sep = " " * 6

        # Widths of left and right columns
        lwidth = (width - len(sep)) // 2
        rwidth = width - (lwidth + len(sep))

        # Function used to convert label-value pair
        # into a sequence of lines within a column
        def align_text(label: str, value: str, position: str) -> typing.List[str]:
            assert position in ["left", "right"]
            if label == "":
                if position == "left":
                    return [f"{value:>{lwidth}}"]
                return [f"{value:>{rwidth}}"]

            # Find width available for the value
            # (+2 stands for colon and space (label: value))
            label_width = len(label) + 2
            if position == "left":
                free_width = lwidth - label_width
            else:
                free_width = rwidth - label_width

            # Split value into chunks using 'free_width'
            value_chunks = [
                value[i : i + free_width] for i in range(0, len(value), free_width)
            ]

            # Collect text row-by-row using 'value_chunks'
            aligned_text = [f"{label}: {value_chunks[0]:>{free_width}}"]
            try:
                for chunk in value_chunks[1:]:
                    aligned_text.append(
                        "".join(
                            [
                                " " * label_width,
                                f"{chunk:>{free_width}}",
                            ]
                        )
                    )
            except IndexError:
                pass
            return aligned_text

        # Function used to convert two label-value pairs
        # into a sequence of rows representing two columns
        def align_pair(
            label: typing.Tuple[str, str],
            value: typing.Tuple[str, str],
        ) -> str:
            parts = [
                align_text(lbl, val, pos)
                for lbl, val, pos in zip(label, value, ("left", "right"))
            ]
            while len(parts[0]) != len(parts[1]):
                shorter_part_index = 0 if len(parts[0]) < len(parts[1]) else 1
                parts[shorter_part_index].append(
                    " " * len(parts[shorter_part_index][0])
                )
            return "\n".join([sep.join([left, right]) for left, right in zip(*parts)])

        # Create summary header
        start_date = (
            f"{calendar.month_name[self.data.index[0].month]} "
            f"{self.data.index[0].year}"
        )
        end_date = (
            f"{calendar.month_name[self.data.index[-1].month]} "
            f"{self.data.index[-1].year}"
        )
        summary = [
            "Univariate Extreme Value Analysis".center(width),
            "=" * width,
            "Source Data".center(width),
            "-" * width,
            align_pair(
                ("Data label", "Size"),
                (str(self.data.name), f"{len(self.data):,d}"),
            ),
            align_pair(
                ("Start", "End"),
                (start_date, end_date),
            ),
            "=" * width,
        ]

        # Fill the extremes section
        summary.extend(
            [
                "Extreme Values".center(width),
                "-" * width,
            ]
        )
        try:
            if self.extremes_method == "BM":
                ev_parameters = (
                    "Block size",
                    str(self.extremes_kwargs["block_size"]),
                )
            elif self.extremes_method == "POT":
                ev_parameters = (
                    "Threshold",
                    str(self.extremes_kwargs["threshold"]),
                )
            else:
                raise AssertionError
            summary.extend(
                [
                    align_pair(
                        ("Count", "Extraction method"),
                        (f"{len(self.extremes):,d}", self.extremes_method),
                    ),
                    align_pair(
                        ("Type", ev_parameters[0]),
                        (self.extremes_type, ev_parameters[1]),
                    ),
                ]
            )
        except AttributeError:
            summary.append("Extreme values have not been extracted")
        summary.append("=" * width)

        # Fill the model section
        summary.extend(
            [
                "Model".center(width),
                "-" * width,
            ]
        )
        try:
            summary.append(
                align_pair(
                    ("Model", "Distribution"),
                    (self.model.name, self.model.distribution.name),
                )
            )
            if self.model.name == "Emcee":
                n_walkers = getattr(self.model, "n_walkers")
                n_samples = getattr(self.model, "n_samples")
                summary.append(
                    align_pair(
                        ("Walkers", "Samples per walker"),
                        (f"{n_walkers:,d}", f"{n_samples:,d}"),
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
            delta_parameters = len(free_parameters) - len(fixed_parameters)
            if delta_parameters < 0:
                for _ in range(-delta_parameters):
                    free_parameters.append("")
            else:
                for _ in range(delta_parameters):
                    fixed_parameters.append("")

            for i, (frp, fip) in enumerate(zip(free_parameters, fixed_parameters)):
                if i == 0:
                    summary.append(
                        align_pair(
                            ("Free parameters", "Fixed parameters"),
                            (frp, fip),
                        )
                    )
                else:
                    summary.append(
                        align_pair(
                            ("", ""),
                            (frp, fip),
                        )
                    )

        except AttributeError:
            summary.append("Model has not been fit to the extremes")

        summary.append("=" * width)

        return "\n".join(summary)

    @typing.overload
    def get_extremes(
        self,
        method: typing.Literal["BM"],
        extremes_type: typing.Literal["high", "low"] = "high",
        *,
        block_size: str = "365.2425D",
        errors: typing.Literal["raise", "ignore", "coerce"] = "raise",
        min_last_block: typing.Optional[float] = None,
    ) -> None:
        ...

    @typing.overload
    def get_extremes(
        self,
        method: typing.Literal["POT"],
        extremes_type: typing.Literal["high", "low"] = "high",
        *,
        threshold: float,
        r: typing.Union[pd.Timedelta, typing.Any] = "24h",
    ) -> None:
        ...

    def get_extremes(
        self,
        method: typing.Literal["BM", "POT"],
        extremes_type: typing.Literal["high", "low"] = "high",
        **kwargs,
    ) -> None:
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
                    Block size (default='365.2425D').
                    See pandas.to_timedelta for more information.
                errors : str, optional
                    raise (default) - raise an exception
                        when encountering a block with no data
                    ignore - ignore blocks with no data
                    coerce - get extreme values for blocks with no data
                        as mean of all other extreme events in the series
                        with index being the middle point of corresponding interval
                min_last_block : float, optional
                    Minimum data availability ratio (0 to 1) in the last block
                    for it to be used to extract extreme value from.
                    This is used to discard last block when it is too short.
                    If None (default), last block is always used.
            if method is POT:
                threshold : float
                    Threshold used to find exceedances.
                r : pandas.Timedelta or value convertible to timedelta, optional
                    Duration of window used to decluster the exceedances.
                    By default r='24H' (24 hours).
                    See pandas.to_timedelta for more information.

        """
        message = f"for method='{method}' and extremes_type='{extremes_type}'"
        logger.debug("extracting extreme values %s", message)
        self.__extremes = get_extremes(
            method=method,
            ts=self.data,
            extremes_type=extremes_type,
            **kwargs,
        )
        self.__extremes_method = method
        self.__extremes_type = extremes_type
        logger.info("successfully extracted extreme values %s", message)

        logger.debug("collecting extreme value properties")
        self.__extremes_kwargs = {}
        if method == "BM":
            self.__extremes_kwargs["block_size"] = pd.to_timedelta(
                kwargs.get("block_size", "365.2425D")
            )
            self.__extremes_kwargs["errors"] = kwargs.get("errors", "raise")
            self.__extremes_kwargs["min_last_block"] = kwargs.get(
                "min_last_block", None
            )
        else:
            self.__extremes_kwargs["threshold"] = kwargs.get("threshold")
            self.__extremes_kwargs["r"] = pd.to_timedelta(kwargs.get("r", "24h"))
        logger.info("successfully collected extreme value properties")

        logger.debug("creating extremes transformer")
        self.__extremes_transformer = ExtremesTransformer(
            extremes=self.__extremes,
            extremes_type=self.__extremes_type,
        )
        logger.info("successfully created extremes transformer")

        logger.info("removing any previously declared models")
        self.__model = None

    @typing.overload
    def set_extremes(
        self,
        extremes: pd.Series,
        method: typing.Literal["BM"] = "BM",
        extremes_type: typing.Literal["high", "low"] = "high",
        *,
        block_size: str = "365.2425D",
        errors: typing.Literal["raise", "ignore", "coerce"] = "raise",
        min_last_block: typing.Optional[float] = None,
    ) -> None:
        ...

    @typing.overload
    def set_extremes(
        self,
        extremes: pd.Series,
        method: typing.Literal["POT"] = "POT",
        extremes_type: typing.Literal["high", "low"] = "high",
        *,
        threshold: float,
        r: typing.Union[pd.Timedelta, typing.Any] = "24h",
    ) -> None:
        ...

    def set_extremes(
        self,
        extremes: pd.Series,
        method: typing.Literal["BM", "POT"] = "BM",
        extremes_type: typing.Literal["high", "low"] = "high",
        **kwargs,
    ) -> None:
        """
        Set extreme values.

        This method is used to set extreme values onto the model instead
        of deriving them from data directly using the 'get_extremes' method.
        This way user can set extremes calculated using a custom methodology.

        Parameters
        ----------
        extremes : pd.Series
            Time series of extreme values to be set onto the model.
            Must be numeric, have date-time index, and have the same name
            as self.data.
        method : str, optional
            Extreme value extraction method.
            Supported values:
                BM (default) - Block Maxima
                POT - Peaks Over Threshold
        extremes_type : str, optional
            high (default) - extreme high values
            low - extreme low values
        kwargs:
            if method is BM:
                block_size : str or pandas.Timedelta, optional
                    Block size.
                    If None (default), then is calculated as median distance
                    between extreme events.
                errors : str, optional
                    raise - raise an exception
                        when encountering a block with no data
                    ignore (default) - ignore blocks with no data
                    coerce - get extreme values for blocks with no data
                        as mean of all other extreme events in the series
                        with index being the middle point of corresponding interval
                min_last_block : float, optional
                    Minimum data availability ratio (0 to 1) in the last block
                    for it to be used to extract extreme value from.
                    This is used to discard last block when it is too short.
                    If None (default), last block is always used.
            if method is POT:
                threshold : float, optional
                    Threshold used to find exceedances.
                    By default is taken as smallest value.
                r : pandas.Timedelta or value convertible to timedelta, optional
                    Duration of window used to decluster the exceedances.
                    By default r='24H' (24 hours).
                    See pandas.to_timedelta for more information.

        """
        # Validate `extremes`
        if not isinstance(extremes, pd.Series):
            raise TypeError(
                f"invalid type in '{type(extremes).__name__}' for the `extremes` "
                f"argument, must be pandas.Series"
            )
        extremes = extremes.copy(deep=True)
        if not isinstance(extremes.index, pd.DatetimeIndex):
            raise TypeError("invalid index type for `extremes`, must be date-time")
        if not np.issubdtype(extremes.dtype, np.number):
            raise TypeError("`extremes` must have numeric values")
        if extremes.name is None:
            extremes.name = self.data.name
        else:
            if extremes.name != self.data.name:
                raise ValueError("`extremes` name doesn't match that of `data`")
        if (
            extremes.index.min() < self.data.index.min()
            or extremes.index.max() > self.data.index.max()
        ):
            raise ValueError("`extremes` time range must fit within that of data")

        # Get `method`
        if method not in ["BM", "POT"]:
            raise ValueError(f"`method` must be either 'BM' or 'POT', not '{method}'")

        # Get `extremes_type`
        if extremes_type not in ["high", "low"]:
            raise ValueError(
                f"`extremes_type` must be either 'BM' or 'POT', not '{extremes_type}'"
            )

        # Get `extremes_kwargs`
        extremes_kwargs = {}
        if method == "BM":
            # Get `block_size`
            extremes_kwargs["block_size"] = pd.to_timedelta(
                kwargs.pop(
                    "block_size",
                    pd.to_timedelta(np.quantile(np.diff(extremes.index), 0.5)),
                )
            )
            if extremes_kwargs["block_size"] <= pd.to_timedelta("0D"):
                raise ValueError(
                    "`block_size` must be a positive timedelta, not %s"
                    % extremes_kwargs["block_size"]
                )

            # Get `errors`
            extremes_kwargs["errors"] = kwargs.pop("errors", "ignore")
            if extremes_kwargs["errors"] not in ["raise", "ignore", "coerce"]:
                raise ValueError(
                    f"invalid value in '{extremes_kwargs['errors']}' "
                    f"for the `errors` argument"
                )

            # Get `min_last_block`
            extremes_kwargs["min_last_block"] = kwargs.pop("min_last_block", None)
            if extremes_kwargs["min_last_block"] is not None:
                if not 0 <= extremes_kwargs["min_last_block"] <= 1:
                    raise ValueError(
                        "`min_last_block` must be a number in the [0, 1] range"
                    )

        else:
            # Get `threshold`
            extremes_kwargs["threshold"] = kwargs.pop(
                "threshold",
                {
                    "high": extremes.min(),
                    "low": extremes.max(),
                }[extremes_type],
            )
            if (
                extremes_type == "high"
                and extremes_kwargs["threshold"] > extremes.values.min()
            ) or (
                extremes_type == "low"
                and extremes_kwargs["threshold"] < extremes.values.max()
            ):
                raise ValueError("invalid `threshold` value")

            # Get `r`
            extremes_kwargs["r"] = pd.to_timedelta(kwargs.pop("r", "24h"))
            if extremes_kwargs["r"] <= pd.to_timedelta("0D"):
                raise ValueError(
                    "`r` must be a positive timedelta, not %s" % extremes_kwargs["r"]
                )

        # Check for unrecognized kwargs
        if len(kwargs) != 0:
            raise TypeError(
                f"unrecognized arguments passed in: {', '.join(kwargs.keys())}"
            )

        # Set attributes
        self.__extremes = extremes
        self.__extremes_method = method
        self.__extremes_type = extremes_type
        self.__extremes_kwargs = extremes_kwargs
        self.__extremes_transformer = ExtremesTransformer(
            extremes=self.__extremes,
            extremes_type=self.__extremes_type,
        )
        self.__model = None
        logger.info("successfully set extremes")

    @typing.overload
    @classmethod
    def from_extremes(
        cls,
        extremes: pd.Series,
        method: typing.Literal["BM"] = "BM",
        extremes_type: typing.Literal["high", "low"] = "high",
        *,
        block_size: str = "365.2425D",
        errors: typing.Literal["raise", "ignore", "coerce"] = "raise",
        min_last_block: typing.Optional[float] = None,
    ) -> None:
        ...

    @typing.overload
    @classmethod
    def from_extremes(
        cls,
        extremes: pd.Series,
        method: typing.Literal["POT"] = "POT",
        extremes_type: typing.Literal["high", "low"] = "high",
        *,
        threshold: float,
        r: typing.Union[pd.Timedelta, typing.Any] = "24h",
    ) -> None:
        ...

    @classmethod
    def from_extremes(
        cls,
        extremes: pd.Series,
        method: typing.Literal["BM", "POT"] = "BM",
        extremes_type: typing.Literal["high", "low"] = "high",
        **kwargs,
    ) -> EVA:
        """
        Create an EVA model using pre-defined `extremes`.

        A typical reason to use this method is when full timeseries is not available
        and only the extracted extremes (i.e. annual maxima) are known.

        Parameters
        ----------
        extremes : pd.Series
            Time series of extreme values.
        method : str, optional
            Extreme value extraction method.
            Supported values:
                BM (default) - Block Maxima
                POT - Peaks Over Threshold
        extremes_type : str, optional
            high (default) - extreme high values
            low - extreme low values
        kwargs:
            if method is BM:
                block_size : str or pandas.Timedelta, optional
                    Block size.
                    If None (default), then is calculated as median distance
                    between extreme events.
                errors : str, optional
                    raise - raise an exception
                        when encountering a block with no data
                    ignore (default) - ignore blocks with no data
                    coerce - get extreme values for blocks with no data
                        as mean of all other extreme events in the series
                        with index being the middle point of corresponding interval
                min_last_block : float, optional
                    Minimum data availability ratio (0 to 1) in the last block
                    for it to be used to extract extreme value from.
                    This is used to discard last block when it is too short.
                    If None (default), last block is always used.
            if method is POT:
                threshold : float, optional
                    Threshold used to find exceedances.
                    By default is taken as smallest value.
                r : pandas.Timedelta or value convertible to timedelta, optional
                    Duration of window used to decluster the exceedances.
                    By default r='24H' (24 hours).
                    See pandas.to_timedelta for more information.

        Returns
        -------
        EVA
            EVA model initialized with `extremes`.

        """
        model = cls(data=extremes)
        model.set_extremes(
            extremes=model.data,
            method=method,
            extremes_type=extremes_type,
            **kwargs,
        )
        return model

    def plot_extremes(
        self,
        figsize: tuple = (8, 5),
        ax: typing.Optional[plt.Axes] = None,
        show_clusters: bool = False,
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
        show_clusters : bool, optional
            If True, show cluster boundaries for POT extremes.
            Has no effect if extremes were extracted using BM method.
            May produce wrong cluster boundaries if extremes were set using the
            `set_extremes` or `from_extremes` methods and threshold and inter-cluster
            distance (r) arguments were not provided.
            By default is False.

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
            r=self.extremes_kwargs.get("r", None) if show_clusters else None,
            figsize=figsize,
            ax=ax,
        )

    @typing.overload
    def fit_model(
        self,
        model: typing.Literal["MLE"] = "MLE",
        distribution: typing.Union[str, scipy.stats.rv_continuous] = None,
        distribution_kwargs: typing.Optional[dict] = None,
    ) -> None:
        ...

    @typing.overload
    def fit_model(
        self,
        model: typing.Literal["Emcee"] = "Emcee",
        distribution: typing.Union[str, scipy.stats.rv_continuous] = None,
        distribution_kwargs: typing.Optional[dict] = None,
        n_walkers: int = 100,
        n_samples: int = 500,
        progress: bool = False,
    ) -> None:
        ...

    def fit_model(
        self,
        model: typing.Literal["MLE", "Emcee"] = "MLE",
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
            Special keyword arguments, passed to the `.fit` method of the distribution.
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
                "selecting default distribution for extremes extracted using the "
                "'%s' method",
                self.extremes_method,
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
            distribution = None
            aic = np.inf
            for distribution_name in candidate_distributions:
                new_aic = MLE(
                    extremes=self.extremes_transformer.transformed_extremes,
                    distribution=distribution_name,
                    distribution_kwargs=_distribution_kwargs,
                ).AIC
                if new_aic < aic:
                    distribution = distribution_name
                    aic = new_aic
            logger.info(
                "selected '%s' distribution with AIC score %s",
                distribution,
                aic,
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
                "freezing location parameter (floc) at %s for '%s' distribution",
                distribution_kwargs["floc"],
                distribution_name,
            )

        # Fit model to transformed extremes
        self.__model = get_model(
            model=model,
            extremes=self.extremes_transformer.transformed_extremes,
            distribution=distribution,
            distribution_kwargs=distribution_kwargs,
            **kwargs,
        )

    def _get_mcmc_plot_inputs(
        self,
        labels: typing.Optional[typing.List[str]] = None,
    ) -> tuple:  # pragma: no cover
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
        labels: typing.Optional[typing.List[str]] = None,
        figsize: typing.Optional[typing.Tuple[float, float]] = None,
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
        labels: typing.Optional[typing.List[str]] = None,
        levels: typing.Optional[int] = None,
        figsize: typing.Tuple[float, float] = (8, 8),
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
            2D list with Axes objects of size N by N, where N is `trace.shape[2]`.
            Empty slots are represented by None. Axes are ordered from left to right
            top to bottom.

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
        return_period,
        return_period_size: typing.Union[str, pd.Timedelta] = "365.2425D",
        alpha: typing.Optional[float] = None,
        **kwargs,
    ) -> tuple:
        """
        Get return value and confidence interval for given return period(s).

        Parameters
        ----------
        return_period : array-like
            Return period or 1D array of return periods.
            Given as a multiple of `return_period_size`.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='365.2425D').
            If set to '30D', then a return period of 12
            would be roughly equivalent to a 1 year return period (360 days).
        alpha : float, optional
            Width of confidence interval (0, 1).
            If None (default), return None
            for upper and lower confidence interval bounds.
        kwargs
            Model-specific keyword arguments.
            If alpha is None, keyword arguments are ignored
            (error still raised for unrecognized arguments).
            MLE model:
                n_samples : int, optional
                    Number of bootstrap samples used to estimate
                    confidence interval bounds (default=100).
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        return_value : array-like
            Return values.
        ci_lower : array-like
            Lower confidence interval bounds.
        ci_upper : array-like
            Upper confidence interval bounds.

        """
        # Parse the 'return_period_size' argument
        if not isinstance(return_period_size, pd.Timedelta):
            if isinstance(return_period_size, str):
                return_period_size = pd.to_timedelta(return_period_size)
            else:
                raise TypeError(
                    f"invalid type in {type(return_period_size)} "
                    f"for the 'return_period_size' argument"
                )

        # Calculate rate of extreme events
        # as number of extreme events per `return_period_size`
        if self.extremes_method == "BM":
            extremes_rate = return_period_size / self.extremes_kwargs["block_size"]
        elif self.extremes_method == "POT":
            n_periods = (self.data.index[-1] - self.data.index[0]) / return_period_size
            extremes_rate = len(self.extremes) / n_periods
        else:
            raise AssertionError

        # Convert 'return_period' to ndarray
        return_period = np.asarray(a=return_period, dtype=np.float64).copy()
        if return_period.ndim == 0:
            return_period = return_period[np.newaxis]
        if return_period.ndim != 1:
            raise ValueError(
                f"invalid shape in {return_period.shape} "
                f"for the 'return_period' argument, must be 1D array"
            )

        # Calculate exceedance probability
        exceedance_probability = 1 / return_period / extremes_rate

        # Calculate return values
        return tuple(
            self.extremes_transformer.transform(value)
            for value in self.model.get_return_value(
                exceedance_probability=exceedance_probability, alpha=alpha, **kwargs
            )
        )

    def get_summary(
        self,
        return_period,
        return_period_size: typing.Union[str, pd.Timedelta] = "365.2425D",
        alpha: typing.Optional[float] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with return values and confidence interval bounds.

        Parameters
        ----------
        return_period : array-like
            Return period or 1D array of return periods.
            Given as a multiple of `return_period_size`.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='365.2425D').
            If set to '30D', then a return period of 12
            would be roughly equivalent to a 1 year return period (360 days).
        alpha : float, optional
            Width of confidence interval (0, 1).
            If None (default), return None
            for upper and lower confidence interval bounds.
        kwargs
            Model-specific keyword arguments.
            If alpha is None, keyword arguments are ignored
            (error still raised for unrecognized arguments).
            MLE model:
                n_samples : int, optional
                    Number of bootstrap samples used to estimate
                    confidence interval bounds (default=100).
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        summary : pandas.DataFrame
            DataFrame with return values and confidence interval bounds.

        """
        # Convert 'return_period' to ndarray
        return_period = np.asarray(a=return_period, dtype=np.float64).copy()
        if return_period.ndim == 0:
            return_period = return_period[np.newaxis]
        if return_period.ndim != 1:
            raise ValueError(
                f"invalid shape in {return_period.shape} "
                f"for the 'return_period' argument, must be 1D array"
            )

        # Calculate return values
        rv = self.get_return_value(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=alpha,
            **kwargs,
        )
        return_values = []
        for value in rv:
            value = np.asarray(a=value, dtype=np.float64)
            if value.ndim == 0:
                value = value[np.newaxis]
            return_values.append(value)

        return pd.DataFrame(
            data=np.transpose(return_values),
            index=pd.Index(data=return_period, name="return period"),
            columns=["return value", "lower ci", "upper ci"],
        )

    def plot_return_values(
        self,
        return_period=None,
        return_period_size: typing.Union[str, pd.Timedelta] = "365.2425D",
        alpha: typing.Optional[float] = None,
        plotting_position: typing.Literal[
            "ecdf",
            "hazen",
            "weibull",
            "tukey",
            "blom",
            "median",
            "cunnane",
            "gringorten",
            "beard",
        ] = "weibull",
        ax: typing.Optional[plt.Axes] = None,
        figsize: typing.Tuple[float, float] = (8, 5),
        **kwargs,
    ) -> tuple:  # pragma: no cover
        """
        Plot return values and confidence intervals for given return periods.

        Parameters
        ----------
        return_period : array-like, optional
            Return period or 1D array of return periods.
            Given as a multiple of `return_period_size`.
            If None (default), calculates as 100 values uniformly spaced
            within the range of return periods of the extracted extreme values.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='365.2425D').
            If set to '30D', then a return period of 12
            would be roughly equivalent to a 1 year return period (360 days).
        alpha : float, optional
            Width of confidence interval (0, 1).
            If None (default), confidence interval bounds are not plotted.
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        ax : matplotlib.axes._axes.Axes, optional
            Axes onto which the return value plot is drawn.
            If None (default), a new figure and axes objects are created.
        figsize : tuple, optional
            Figure size in inches in format (width, height).
            By default it is (8, 5).
        kwargs
            Model-specific keyword arguments.
            If alpha is None, keyword arguments are ignored
            (error still raised for unrecognized arguments).
            MLE model:
                n_samples : int, optional
                    Number of bootstrap samples used to estimate
                    confidence interval bounds (default=100).
            Emcee model:
                burn_in : int
                    Burn-in value (number of first steps to discard for each walker).

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : matplotlib.axes._axes.Axes
            Axes object.

        """
        # Get observed return values
        observed_return_values = get_return_periods(
            ts=self.data,
            extremes=self.extremes,
            extremes_method=self.extremes_method,
            extremes_type=self.extremes_type,
            block_size=self.extremes_kwargs.get("block_size", None),
            return_period_size=return_period_size,
            plotting_position=plotting_position,
        )

        # Parse the 'return_period' argument
        if return_period is None:
            return_period = np.linspace(
                observed_return_values.loc[:, "return period"].min(),
                observed_return_values.loc[:, "return period"].max(),
                100,
            )
        else:
            # Convert 'return_period' to ndarray
            return_period = np.asarray(a=return_period, dtype=np.float64).copy()
            if return_period.ndim == 0:
                return_period = return_period[np.newaxis]
            if return_period.ndim != 1:
                raise ValueError(
                    f"invalid shape in {return_period.shape} "
                    f"for the 'return_period' argument, must be 1D array"
                )
            if len(return_period) < 2:
                raise ValueError(
                    f"'return_period' must have at least 2 return periods, "
                    f"{len(return_period)} was given"
                )

        # Get modeled return values
        modeled_return_values = self.get_summary(
            return_period=return_period,
            return_period_size=return_period_size,
            alpha=alpha,
            **kwargs,
        )

        # Plot return values
        return plot_return_values(
            observed_return_values=observed_return_values,
            modeled_return_values=modeled_return_values,
            ax=ax,
            figsize=figsize,
        )

    def plot_probability(
        self,
        plot_type: str,
        return_period_size: typing.Union[str, pd.Timedelta] = "365.2425D",
        plotting_position: typing.Literal[
            "ecdf",
            "hazen",
            "weibull",
            "tukey",
            "blom",
            "median",
            "cunnane",
            "gringorten",
            "beard",
        ] = "weibull",
        ax: typing.Optional[plt.Axes] = None,
        figsize: typing.Tuple[float, float] = (8, 8),
    ) -> tuple:  # pragma: no cover
        """
        Plot a probability plot (QQ or PP).

        Parameters
        ----------
        plot_type : str
            Probability plot type.
            Supported values:
                PP - probability plot
                QQ - quantile plot
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='365.2425D').
            If set to '30D', then a return period of 12
            would be roughly equivalent to a 1 year return period (360 days).
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        ax : matplotlib.axes._axes.Axes, optional
            Axes onto which the probability plot is drawn.
            If None (default), a new figure and axes objects are created.
        figsize : tuple, optional
            Figure size in inches in format (width, height).
            By default it is (8, 8).

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure object.
        axes : matplotlib.axes._axes.Axes
            Axes object.

        """
        # Get observed return values
        observed_return_values = get_return_periods(
            ts=self.data,
            extremes=self.extremes,
            extremes_method=self.extremes_method,
            extremes_type=self.extremes_type,
            block_size=self.extremes_kwargs.get("block_size", None),
            return_period_size=return_period_size,
            plotting_position=plotting_position,
        )

        # Get observed and theoretical values
        # depending on 'plot_type'
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
                f"invalid value in '{plot_type}' for the 'plot_type' argument, "
                f"available values: PP, QQ"
            )

        # Plot the probability plot
        return plot_probability(
            observed=observed,
            theoretical=theoretical,
            ax=ax,
            figsize=figsize,
        )

    def plot_diagnostic(
        self,
        return_period=None,
        return_period_size: typing.Union[str, pd.Timedelta] = "365.2425D",
        alpha: typing.Optional[float] = None,
        plotting_position: typing.Literal[
            "ecdf",
            "hazen",
            "weibull",
            "tukey",
            "blom",
            "median",
            "cunnane",
            "gringorten",
            "beard",
        ] = "weibull",
        figsize: typing.Tuple[float, float] = (8, 8),
        **kwargs,
    ):  # pragma: no cover
        """
        Plot a diagnostic plot.

        This plot shows four key plots characterizing the EVA model:
            - top left : return values plot
            - top right : probability density (PDF) plot
            - bottom left : quantile (Q-Q) plot
            - bottom right : probability (P-P) plot

        Parameters
        ----------
        return_period : array-like, optional
            Return period or 1D array of return periods.
            Given as a multiple of `return_period_size`.
            If None (default), calculates as 100 values uniformly spaced
            within the range of return periods of the extracted extreme values.
        return_period_size : str or pandas.Timedelta, optional
            Size of return periods (default='365.2425D').
            If set to '30D', then a return period of 12
            would be roughly equivalent to a 1 year return period (360 days).
        alpha : float, optional
            Width of confidence interval (0, 1).
            If None (default), confidence interval bounds are not plotted.
        plotting_position : str, optional
            Plotting position name (default='weibull'), not case-sensitive.
            Supported plotting positions:
                ecdf, hazen, weibull, tukey, blom, median, cunnane, gringorten, beard
        figsize : tuple, optional
            Figure size in inches in format (width, height).
            By default it is (8, 8).
        kwargs
            Model-specific keyword arguments.
            If alpha is None, keyword arguments are ignored
            (error still raised for unrecognized arguments).
            MLE model:
                n_samples : int, optional
                    Number of bootstrap samples used to estimate
                    confidence interval bounds (default=100).
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
            # Create figure
            fig = plt.figure(figsize=figsize, dpi=96)

            # Create gridspec
            gs = matplotlib.gridspec.GridSpec(
                nrows=2,
                ncols=2,
                wspace=0.3,
                hspace=0.3,
                width_ratios=[1, 1],
                height_ratios=[1, 1],
            )

            # Create axes
            ax_rv = fig.add_subplot(gs[0, 0])
            ax_pdf = fig.add_subplot(gs[0, 1])
            ax_qq = fig.add_subplot(gs[1, 0])
            ax_pp = fig.add_subplot(gs[1, 1])

            # Plot return values
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

            # Plot PDF
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
                color="k",
                lw=0.5,
                zorder=15,
            )
            ax_pdf.set_ylim(0, ax_pdf.get_ylim()[1])

            # Plot Q-Q plot
            self.plot_probability(
                plot_type="QQ",
                return_period_size=return_period_size,
                plotting_position=plotting_position,
                ax=ax_qq,
            )
            ax_qq.set_title("Q-Q plot")

            # Plot P-P plot
            self.plot_probability(
                plot_type="PP",
                return_period_size=return_period_size,
                plotting_position=plotting_position,
                ax=ax_pp,
            )
            ax_pp.set_title("P-P plot")

            return fig, (ax_rv, ax_pdf, ax_qq, ax_pp)
