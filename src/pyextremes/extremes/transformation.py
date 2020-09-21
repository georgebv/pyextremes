import typing

import pandas as pd


class ExtremesTransformer:

    __slots__ = [
        "extremes",
        "extremes_type",
        "transformed_extremes",
    ]

    def __init__(self, extremes: pd.Series, extremes_type: str) -> None:
        """
        Extreme value transformer class.

        Provides methods to transform extreme value series to and from
        format compatible with pyextremes models.
        Values are transformed in a way to have extreme events at the right side
        of a distribution support.
        For extremes_type being 'high' no transform is performed.
        For extremes_type being 'low' values are flipped around the maximum value.

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
        self.transformed_extremes: pd.Series = self.transform(value=extremes)

    def transform(
        self, value: typing.Union[None, float, pd.Series]
    ) -> typing.Union[None, float, pd.Series]:
        """
        Transform extreme values.

        Works both ways.

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
        if value is None:
            return value
        else:
            if self.extremes_type == "high":
                return value
            else:
                return 2 * self.extremes.max() - value
