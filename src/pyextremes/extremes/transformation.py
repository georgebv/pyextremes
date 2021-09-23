import typing

import pandas as pd


class ExtremesTransformer:
    """
    Utility class used to transform extreme values between high and low extremes type.

    Provides methods to transform extreme value series to and from format
    compatible with pyextremes models. Extreme values are transformed in a way
    to have more extreme events (lower probability of exceedance) oriented to the right
    (in the increasing order).
    For extremes_type being 'high' no transform is performed.
    For extremes_type being 'low' values are flipped around the maximum value.

    """

    __slots__ = [
        "extremes",
        "extremes_type",
        "__pivot_point",
        "__transformed_extremes",
    ]

    def __init__(self, extremes: pd.Series, extremes_type: str) -> None:
        """
        Initialize the extreme value transformer.

        Parameters
        ----------
        extremes : pandas.Series
            Time series of extreme events.
        extremes_type : str
            high - provided extreme values are extreme high values
            low - provided extreme values are extreme low values

        """
        self.extremes: pd.Series = extremes
        self.extremes_type: str = extremes_type

        self.__pivot_point: typing.Optional[float] = (
            None if self.extremes_type == "high" else self.extremes.max()
        )
        self.__transformed_extremes: pd.Series = self.transform(value=extremes)

    @property
    def pivot_point(self) -> typing.Optional[float]:
        return self.__pivot_point

    @property
    def transformed_extremes(self) -> pd.Series:
        return self.__transformed_extremes

    def transform(
        self, value: typing.Union[None, float, pd.Series]
    ) -> typing.Union[None, float, pd.Series]:
        """
        Transform extreme values in either direction.

        Parameters
        ----------
        value : float or pandas.Series
            Value or series of values to be transformed.

        Returns
        -------
        transformed_value : None, float, or pandas.Series
            Transformed value or series.
            If value is None, returns None (a.k.a. null-transform).

        """
        if value is None or self.pivot_point is None:
            return value
        return 2 * self.pivot_point - value
