The final goal of most extreme value analyses is estimating **return values** -
levels which are expected to be exceeded on average once every given return period
(e.g. the 100-year water level). Once a model is fitted
(see [Fitting Models](7-models.md)), return values are calculated from the
fitted distribution for any return period, including periods much longer than
the observed record.

## Getting Return Values

Return values are calculated using the `get_return_value` method:

=== "Code"

    ```python
    from pyextremes import EVA

    model = EVA(data=data)
    model.get_extremes("BM", block_size="365.2425D")
    model.fit_model()

    return_value, lower_ci, upper_ci = model.get_return_value(return_period=100)
    print(return_value, lower_ci, upper_ci)
    ```

=== "Output"

    ```
    2.671312557797694 nan nan
    ```

??? note
    You can get the `data` variable referenced above by running the following code:

    ```python
    import pandas as pd

    data = pd.read_csv(
        "battery_wl.csv",
        index_col=0,
        parse_dates=True,
    ).squeeze()
    data = (
        data
        .sort_index(ascending=True)
        .astype(float)
        .dropna()
        .loc[pd.to_datetime("1925"):]
    )
    data = (
        data
        - (data.index.array - pd.to_datetime("1992"))
        / pd.to_timedelta("365.2425D")
        * 2.87e-3
    )
    ```

    `#!python "battery_wl.csv"`
    [can be downloaded here](https://github.com/georgebv/pyextremes-notebooks/tree/master/data).

    All figures shown in the Results sections were generated using
    the code shown in these sections and the `data` defined above.

The method always returns a tuple of three values: the return value and
the lower and upper confidence interval bounds. Confidence intervals are `nan`
unless the `alpha` argument is provided (see
[Confidence Intervals](13-confidence-intervals.md)).
If `return_period` is an array, each of the three values is an array as well.

The `get_return_value` method uses the following parameters:

- **return_period** - return period or 1D array of return periods,
  given as multiples of `return_period_size`.
- **return_period_size** - size of return period as a string or `pandas.Timedelta`.
  By default this is 1 year (`#!python "365.2425D"`).
- **alpha** - confidence interval width in the range (0, 1).
  If None (default), confidence intervals are not calculated.
- **kwargs** - model-specific arguments used to calculate confidence intervals:
  `n_samples` for `MLE`, `MOM`, and `Lmoments` models and `burn_in` for the
  `Emcee` model. Ignored if `alpha` is None.

## Summary Table

The `get_summary` method takes the same arguments and returns
a `pandas.DataFrame`, which is more convenient for tabulating results:

=== "Code"

    ```python
    summary = model.get_summary(return_period=[2, 10, 100])
    print(summary)
    ```

=== "Output"

    ```
                   return value  lower ci  upper ci
    return period
    2.0                1.409343       NaN       NaN
    10.0               1.803499       NaN       NaN
    100.0              2.671313       NaN       NaN
    ```

The same values can be visualized using the `plot_return_values` method,
which also shows the observed extreme values at their
[empirical return periods](6-return-periods.md).
If `return_period` is not provided, the plot covers the range of
return periods of the observed extreme values.
See [Maximum Likelihood](9-mle.md#confidence-intervals) for an example.

## Return Period Size

Return periods are multiples of `return_period_size`, which is one year
(`#!python "365.2425D"`) by default. Any other duration can be used,
for example, to work with return periods measured in months:

=== "Code"

    ```python
    summary = model.get_summary(
        return_period=[24, 120, 1200],
        return_period_size="30D",
    )
    print(summary)
    ```

=== "Output"

    ```
                   return value  lower ci  upper ci
    return period
    24.0               1.405976       NaN       NaN
    120.0              1.799451       NaN       NaN
    1200.0             2.664111       NaN       NaN
    ```

A 1200-month (~98.6-year) water level is, as expected, close to the 100-year one.

!!! warning "Valid return periods"
    Return values can only be calculated for return periods longer than the
    average interval between extreme events. Annual maxima occur once every
    ~12.2 months, so return periods of 6 or 12 months (with
    `#!python return_period_size="30D"`) are too short and their return values are
    `-inf`. Similarly, for POT extremes the shortest valid return period depends
    on how many extreme events were extracted
    (see [Maximum Likelihood](9-mle.md#confidence-intervals)).

## How Return Values are Calculated

A return period $R$ (in multiples of `return_period_size`) is converted to the
probability $p$ of a single extreme event exceeding the return value:

$$p = \frac{1}{R \lambda}$$

where $\lambda$ is the average number of extreme events per `return_period_size`
(see [Estimating Return Periods](6-return-periods.md#return-period)).
The return value is then the value which is exceeded with probability $p$
according to the fitted distribution, i.e. its inverse survival function
(`isf`). For annual maxima $\lambda = 1$, so the 100-year water level is the value
exceeded by an annual maximum with probability 1%.

Models created using the standalone `get_model` function know nothing about
the time series or return periods, so they take the exceedance probability
directly:

=== "Code"

    ```python
    from pyextremes import get_extremes, get_model

    extremes = get_extremes(data, "BM", block_size="365.2425D")
    model = get_model(model="MLE", extremes=extremes, distribution="genextreme")
    print(model.get_return_value(exceedance_probability=0.01))
    ```

=== "Output"

    ```
    (np.float64(2.671312557797694), np.float64(nan), np.float64(nan))
    ```

!!! tip
    When using the standalone functions with Peaks Over Threshold extremes,
    calculate $\lambda$ as the number of extreme values divided by the duration
    of the time series in units of `return_period_size`.
