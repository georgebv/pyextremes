In addition to [MLE](9-mle.md) and [MCMC](10-mcmc.md), `pyextremes` supports
two moment-based fitting methods: the Method of Moments (`MOM`) and L-moments
(`Lmoments`). Both estimate distribution parameters by matching summary statistics
of the distribution to those of the observed extreme values, instead of maximizing
the likelihood.

## Method of Moments

The [Method of Moments](https://en.wikipedia.org/wiki/Method_of_moments_(statistics))
finds distribution parameters for which the theoretical moments of the distribution
(mean, variance, skewness, etc.) are equal to the sample moments
of the extreme values. A distribution with $k$ free parameters is fitted by matching
its first $k$ moments. `pyextremes` uses the `fit` method of
[scipy.stats.rv_continuous](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html)
with `#!python method="mm"`. Like MLE, it works with any continuous distribution
and supports fixing distribution parameters.

=== "Standalone"

    ```python
    from pyextremes import get_extremes, get_model

    extremes = get_extremes(data, "BM", block_size="365.2425D")
    model = get_model(
        model="MOM",
        extremes=extremes,
        distribution="gumbel_r",
    )
    ```

=== "Using EVA"

    ```python
    from pyextremes import EVA

    model = EVA(data=data)
    model.get_extremes("BM", block_size="365.2425D")
    model.fit_model(model="MOM", distribution="gumbel_r")
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

!!! warning "Heavy tails"
    Moments of heavy-tailed distributions may not exist. For example,
    the skewness of the GEVD is infinite for $\xi \geq 1/3$ and the variance
    is infinite for $\xi \geq 1/2$. Even when the true distribution has finite moments,
    the optimizer may reach such parameter values and fail. This is what happens
    when fitting `genextreme` to the Battery annual maxima, which are heavy-tailed
    ($\xi \approx 0.27$):

    ```
    ValueError: Method of moments encountered a non-finite distribution moment and
    cannot continue. Consider trying method='MLE'.
    ```

    Because rare events are governed by the tail of the distribution, MOM is
    generally not recommended for EVA, except for distributions with
    no shape parameter (e.g. `gumbel_r` or `expon`), where it is simple and robust.

## L-moments

[L-moments](https://en.wikipedia.org/wiki/L-moment) are analogues of
conventional moments calculated as linear combinations of ordered data values.
Compared with conventional moments, they exist whenever the mean exists,
are less sensitive to outliers, and produce reliable estimates from small samples,
which has made them a standard tool in hydrology and regional frequency
analysis (Hosking, 1990). `pyextremes` uses the
[lmoments3](https://github.com/Ouranosinc/lmoments3) package, which must be
installed separately:

```shell
pip install "pyextremes[lmoments]"
```

=== "Standalone"

    ```python
    from pyextremes import get_extremes, get_model

    extremes = get_extremes(data, "BM", block_size="365.2425D")
    model = get_model(
        model="Lmoments",
        extremes=extremes,
        distribution="genextreme",
    )
    ```

=== "Using EVA"

    ```python
    from pyextremes import EVA

    model = EVA(data=data)
    model.get_extremes("BM", block_size="365.2425D")
    model.fit_model(model="Lmoments", distribution="genextreme")
    ```

Unlike MOM, L-moments handle the heavy-tailed annual maxima without issues
and produce parameters very close to MLE:

=== "Code"

    ```python
    print(model.model)
    ```

=== "Output"

    ```
                     Lmoments model
    -------------------------------------------------
    free parameters: c=-0.260, loc=1.352, scale=0.147
    fixed parameters: all parameters are free
    AIC: -29.758
    loglikelihood: 18.012
    return value cache size: 0
    fit parameter cache size: 0
    -------------------------------------------------
    ```

The `Lmoments` model has the following limitations:

- Only the following distributions are supported (referred to by their `scipy.stats`
  names and passed as strings): `expon`, `gamma`, `genextreme`, `genlogistic`,
  `gennorm`, `genpareto`, `gumbel_r`, `kappa4`, `norm`, `pearson3`, and `weibull_min`.
- Distribution parameters cannot be fixed. Because of this, `EVA` doesn't fix the
  location parameter of `genpareto` and `expon` distributions at the threshold
  (see [Distributions](8-distributions.md#location-parameter-in-pot)) and
  the location parameter is fitted instead. For example, for the POT extremes
  above a threshold of 1.5 m the fitted location is 1.506 m.

## Comparison

Comparing the fitting methods is a good way to check how sensitive the results are
to the choice of method. The figure below shows the Gumbel distribution fitted to the
annual maxima using MLE, MOM, and L-moments
(produced using the same code as the figures in
[Distributions](8-distributions.md#default-distribution),
with `#!python fit_model(model=..., distribution="gumbel_r")`):

<figure>
  <img src="../../img/models/bm-gumbel-methods.png" alt="Gumbel fitted using MLE, MOM, and L-moments"/>
</figure>

| Distribution | Model      | 100-year water level, m |
| :----------- | :--------- | :---------------------: |
| `genextreme` | `MLE`      | 2.671                   |
| `genextreme` | `Lmoments` | 2.658                   |
| `genextreme` | `MOM`      | fails (see above)       |
| `gumbel_r`   | `MLE`      | 2.161                   |
| `gumbel_r`   | `Lmoments` | 2.292                   |
| `gumbel_r`   | `MOM`      | 2.421                   |

When the distribution describes the data well (`genextreme`), different methods
agree closely. When it doesn't (`gumbel_r` can't represent the heavy tail of these
data), the methods disagree, with MOM being the most influenced by the largest
observations. A large disagreement between methods is a sign that the
selected distribution may not be appropriate.

!!! info
    AIC should not be used to compare models fitted using different methods.
    MLE maximizes the likelihood by definition, so it always has the smallest AIC
    for a given distribution. Use AIC to compare distributions fitted using
    the same method (see [Distributions](8-distributions.md#default-distribution)).

Confidence intervals for both `MOM` and `Lmoments` are estimated using bootstrap
resampling with the `n_samples` argument, as described in
[Maximum Likelihood](9-mle.md#confidence-intervals).
