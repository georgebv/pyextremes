This section demonstrates how empirical probabilities (return periods) can be obtained
for extreme values extracted using methods described in earlier sections.

## What is Return Period
Return period indicates duration of time (typically years) which corresponds to a
probability that a given value (e.g. wind speed) would be exceeded at least once within
a year. This probability is called probability of exceedance and is related to return
periods as `1/p` where `p` is return period.

!!! quote "Coles (2001, p.49)"
    In common terminology, $z_{p}$ is the **return level** associated with the
    **return period** $1/p$, since to a reasonable degree of accuracy,
    the level $z_{p}$ is expected to be exceeded on average once every $1/p$ years.
    More precisely, $z_{p}$ is exceeded by the annual maximum in any particular
    year with probability $p$.

Return periods are often incorrectly interpreted in the professional communities
as "100-year event is an event which happens only once in 100 years", which may lead to
inaccurate assessment of risks. A more holistic way of looking at this is to consider
a time period within which a risk is evaluated. For example, a 100-year event with
probability of exceedance in any given year of 1% would have a probability of ~39.5%
to be exceeded at least once within 50 years - this is calculated using this formula:

$$1 - (1 - p) ^ n$$

Where $n$ is number of return period blocks within a time period
(50 for 50 years with retun period block of size 1 year) and $p$ is 1% (100-year event).

## Empirical Return Periods
Empirical return periods are assigned to observed extreme values using an empricial rule
where extreme values are ordered, ranked from most extreme (1) to least extreme (n), and
assigned probabilities using the following formula:

$$P = \frac{r - \alpha}{n + 1 - \alpha - \beta}$$

where:

- **r** - rank of extreme value (1 to n). In `pyextremes` rank is calculated using
  `scipy.stats.rankdata` with `#!python method="average"`, which means that extreme
  events of the same magnitude are assigned average of ranks these values would be
  assigned otherwise if ranked sequentially. For example, array of `[1, 2, 3, 3, 4]`
  would have ranks of `[5, 4, 2.5, 2.5, 1]`.
- **n** - number of extreme values.
- $\alpha$ and $\beta$ - empricial plotting position parameters (see further below).

### Plotting Positions
Plotting positions are sets of empirical coefficients defining how extreme values are
assigned probabilities, which are subsequently used to plot extreme values on the
probability plots.

!!! warning
    Plotting positions have nothing to do with modeling extreme event statistics in
    modern EVA. Historically, in time before computers became widespread, EVA
    was performed by plotting extreme events on probability paper (with axes scaled
    logarithmically and according to a specific plotting position) with the idea that
    a return value curve for a given model (e.g. GEVD) would be a straight line drawn
    through these points using a pen and a ruler.

    Modern EVA fits models to data by maximimizng likelihood function via methods such
    as MLE or MCMC (read more in other sections). This is only feasible due to the use
    of computers and would be prohibitively expensive to do manually. Plotting positions
    are presently used only to show extreme values on return value plots and to
    perform some goodness-of-fit tests (e.g. P-P or Q-Q plots).

    TL;DR: plotting positions are **NOT** used to fit models.

An example of plotting positions used in `pyextremes` is the diagnostic plot where
observed extreme values (black dots) are superimposed against the theoretical
estimates (by fitting a distribution) as seen in the return value, Q-Q, and P-P plots.

<figure>
  <img src="../../img/extremes/selected-threshold-diagnostic.png" alt="Diagnostic plot"/>
</figure>

### Estimating Return Periods
`pyextremes` estimates empirical return periods for many plotting functions and
goodness-of-fit tests behind the scenes using the Weibull plotting position.
Return periods can be calculated using the `get_return_periods` function:

=== "weibull (default)"

    ```python
    from pyextremes import get_extremes, get_return_periods

    extremes = get_extremes(
        ts=data,
        method="BM",
        block_size="365.2425D",
    )
    return_periods = get_return_periods(
        ts=data,
        extremes=extremes,
        extremes_method="BM",
        extremes_type="high",
        block_size="365.2425D",
        return_period_size="365.2425D",
        plotting_position="weibull",
    )
    return_periods.sort_values("return period", ascending=False).head()
    ```

    |   Date-Time (GMT)   | Water Elevation [m NAVD88] | exceedance probability | return period |
    | :-----------------: | :------------------------: | :--------------------: | :-----------: |
    | 2012-10-30 01:00:00 |          3.357218          |        0.010526        |   95.000000   |
    | 1960-09-12 18:00:00 |          2.295832          |        0.021053        |   47.500000   |
    | 1992-12-11 14:00:00 |          2.108284          |        0.031579        |   31.666667   |
    | 1953-11-07 12:00:00 |          2.101487          |        0.042105        |   23.750000   |
    | 1950-11-25 14:00:00 |          2.012957          |        0.052632        |   19.000000   |

=== "median"

    ```python
    from pyextremes import get_extremes, get_return_periods

    extremes = get_extremes(
        ts=data,
        method="BM",
        block_size="365.2425D",
    )
    return_periods = get_return_periods(
        ts=data,
        extremes=extremes,
        extremes_method="BM",
        extremes_type="high",
        block_size="365.2425D",
        return_period_size="365.2425D",
        plotting_position="median",
    )
    return_periods.sort_values("return period", ascending=False).head()
    ```

    |   Date-Time (GMT)   | Water Elevation [m NAVD88] | exceedance probability | return period |
    | :-----------------: | :------------------------: | :--------------------: | :-----------: |
    | 2012-10-30 01:00:00 |          3.357218          |        0.007233        |  138.263736   |
    | 1960-09-12 18:00:00 |          2.295832          |        0.017830        |   56.086181   |
    | 1992-12-11 14:00:00 |          2.108284          |        0.028427        |   35.178006   |
    | 1953-11-07 12:00:00 |          2.101487          |        0.039024        |   25.625255   |
    | 1950-11-25 14:00:00 |          2.012957          |        0.049621        |   20.152696   |

=== "cunnane"

    ```python
    from pyextremes import get_extremes, get_return_periods

    extremes = get_extremes(
        ts=data,
        method="BM",
        block_size="365.2425D",
    )
    return_periods = get_return_periods(
        ts=data,
        extremes=extremes,
        extremes_method="BM",
        extremes_type="high",
        block_size="365.2425D",
        return_period_size="365.2425D",
        plotting_position="cunnane",
    )
    return_periods.sort_values("return period", ascending=False).head()
    ```

    |   Date-Time (GMT)   | Water Elevation [m NAVD88] | exceedance probability | return period |
    | :-----------------: | :------------------------: | :--------------------: | :-----------: |
    | 2012-10-30 01:00:00 |          3.357218          |        0.006369        |  157.000000   |
    | 1960-09-12 18:00:00 |          2.295832          |        0.016985        |   58.875000   |
    | 1992-12-11 14:00:00 |          2.108284          |        0.027601        |   36.230769   |
    | 1953-11-07 12:00:00 |          2.101487          |        0.038217        |   26.166667   |
    | 1950-11-25 14:00:00 |          2.012957          |        0.048832        |   20.478261   |

=== "gringorten"

    ```python
    from pyextremes import get_extremes, get_return_periods

    extremes = get_extremes(
        ts=data,
        method="BM",
        block_size="365.2425D",
    )
    return_periods = get_return_periods(
        ts=data,
        extremes=extremes,
        extremes_method="BM",
        extremes_type="high",
        block_size="365.2425D",
        return_period_size="365.2425D",
        plotting_position="gringorten",
    )
    return_periods.sort_values("return period", ascending=False).head()
    ```

    |   Date-Time (GMT)   | Water Elevation [m NAVD88] | exceedance probability | return period |
    | :-----------------: | :------------------------: | :--------------------: | :-----------: |
    | 2012-10-30 01:00:00 |          3.357218          |        0.005950        |  168.071429   |
    | 1960-09-12 18:00:00 |          2.295832          |        0.016575        |   60.333333   |
    | 1992-12-11 14:00:00 |          2.108284          |        0.027199        |   36.765625   |
    | 1953-11-07 12:00:00 |          2.101487          |        0.037824        |   26.438202   |
    | 1950-11-25 14:00:00 |          2.012957          |        0.048449        |   20.640351   |

The `get_return_periods` function uses the following parameters:

- **ts** - time series (`pandas.Series`) from which the extreme values are extracted
- **extremes** - time series of extreme values.
- **extremes_method** - extreme value extraction method, must be `#!python "BM"` or
  `#!python "POT"`.
- **extremes_type** - extreme value type:
  `#!python "high"` for above threshold (default)
  and `#!python "low"` for below threshold.
- **return_period_size** - size of return period. Same as the `r` argument.
  By default this is 1 year.
- **plotting_position** : plotting position name, case-insensitive.
  Supported plotting positions: ecdf, hazen, weibull (default), tukey, blom, median,
  cunnane, gringorten, beard.

The following paramters are used only when `#!python extremes_method="BM"`:

- **block_size** - block size, by default `#!python "365.2425D"`.
  Internally is converted using the `#!python pandas.to_timedelta` function.
  If not provided, then it is calculated as median distance between extreme values.

??? note
    You can get the `data` variable referenced above by running the following code:

    ```python
    data = pd.read_csv(
        "battery_wl.csv",
        index_col=0,
        parse_dates=True,
        squeeze=True,
    )
    data = (
        data
        .sort_index(ascending=True)
        .astype(float)
        .dropna()
        .loc[pd.to_datetime("1925"):]
    )
    data = (
      data - (data.index.array - pd.to_datetime("1992"))
    ) / pd.to_timedelta("365.2425D") * 2.87e-3
    ```

    `#!python "battery_wl.csv"`
    [can be downloaded here](https://github.com/georgebv/pyextremes-notebooks/tree/master/data).

    All figures shown in this tutorial section were generated using
    [this jupyter notebook](https://nbviewer.jupyter.org/github/georgebv/pyextremes-notebooks/blob/master/notebooks/documentation/extremes/5%20return%20periods.ipynb).
