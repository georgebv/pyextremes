All `pyextremes` models are built around the `Distribution` class
(`pyextremes.models.Distribution`). It wraps a
[scipy.stats.rv_continuous](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html)
distribution and keeps track of which of its parameters are free (fitted) and
which are fixed. Understanding this class is useful when building custom workflows
on top of `pyextremes` or when using custom distributions.

## Creating a Distribution

A distribution is created from extreme values, a distribution name or object,
a fit method, and optional fixed parameters (named with the `f` prefix,
see [Distributions](../user-guide/8-distributions.md#fixing-parameters)).
The distribution is fitted to the extreme values when it is created:

=== "All parameters free"

    ```python
    from pyextremes import get_extremes
    from pyextremes.models import Distribution

    extremes = get_extremes(data, "BM", block_size="365.2425D")
    distribution = Distribution(
        extremes=extremes,
        distribution="genextreme",
        fit_method="MLE",
    )
    print(distribution)
    ```

    ```
                  pyextremes distribution
    ---------------------------------------------------
    name: genextreme
    free parameters: c, loc, scale
    fixed parameters: all parameters are free
    fitted parameters: c=-0.266, loc=1.353, scale=0.146
    ---------------------------------------------------
    ```

=== "Fixed location"

    ```python
    distribution = Distribution(
        extremes=extremes,
        distribution="genpareto",
        floc=1.0,
    )
    print(distribution)
    ```

    ```
            pyextremes distribution
    ----------------------------------------
    name: genpareto
    free parameters: c, scale
    fixed parameters: floc=1.000
    fitted parameters: c=-0.187, scale=0.564
    ----------------------------------------
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

The `fit_method` argument is one of `#!python "MLE"` (default), `#!python "MOM"`,
or `#!python "Lmoments"` and corresponds to the models with the same names.
Invalid fixed parameters (e.g. `fshape` instead of `fc`) raise a `TypeError`
listing the valid names, and fixing all parameters raises a `ValueError`.

A fitted model exposes its distribution as the `distribution` attribute
(`model.distribution` for both `EVA` and models returned by `get_model`).

## Attributes

| Attribute                 | Description                                                        | Example                         |
| :------------------------ | :----------------------------------------------------------------- | :------------------------------ |
| `name`                    | distribution name                                                  | `#!python "genextreme"`         |
| `distribution`            | underlying `scipy.stats.rv_continuous` object                      | `scipy.stats.genextreme`        |
| `distribution_parameters` | all parameter names, in the order used by `scipy.stats`            | `#!python ["c", "loc", "scale"]` |
| `free_parameters`         | names of parameters which are fitted                               | `#!python ["c", "loc", "scale"]` |
| `fixed_parameters`        | fixed parameters with the `f` prefix                               | `#!python {}`                   |
| `number_of_parameters`    | number of free parameters (used to calculate AIC)                  | `#!python 3`                    |
| `fit_method`              | fit method                                                         | `#!python "MLE"`                |
| `mle_parameters`          | fitted values of free parameters (using `fit_method`, despite the name) | `#!python {"c": -0.266, ...}` |

## Methods

### log_probability

Calculates the log-probability of free parameter values `theta` given the extreme
values. It is used by the `Emcee` model as the log-posterior function and is equal
to the log-likelihood, because `pyextremes` uses a flat prior
(see [Markov Chain Monte Carlo](../user-guide/10-mcmc.md)).
Parameter values for which an extreme value falls outside the support of the
distribution have zero probability, so the log-probability is `-inf`:

```python
distribution.log_probability(theta=(-0.266, 1.353, 0.146))  # 18.026
distribution.log_probability(theta=(-0.266, 3.0, 0.146))  # -inf
```

### free2full_parameters

Converts free parameter values (a dictionary, a 1D array, or a 2D array with one
set of parameters per row) to arrays of all distribution parameters, in the
order of `distribution_parameters`, with fixed values filled in:

```python
distribution = Distribution(extremes=extremes, distribution="genpareto", floc=1.0)
distribution.free2full_parameters(free_parameters={"c": 0.1, "scale": 0.2})
# array([0.1, 1. , 0.2])
distribution.free2full_parameters(free_parameters=[[0.1, 0.2], [0.3, 0.4]])
# array([[0.1, 1. , 0.2],
#        [0.3, 1. , 0.4]])
```

### get_prop

Evaluates a property of the `scipy.stats` distribution (e.g. `pdf`, `logpdf`, `cdf`,
`ppf`, or `isf`) for given values and free parameters. Passing a 2D array of free
parameters evaluates the property for many parameter sets at once, which is how
`Emcee` calculates return values for every posterior sample:

```python
distribution = Distribution(extremes=extremes, distribution="genextreme")

# 100-year return value for the fitted parameters
distribution.get_prop(prop="isf", x=0.01, free_parameters=distribution.mle_parameters)
# 2.671

# 10- and 100-year return values
distribution.get_prop(prop="isf", x=[0.1, 0.01], free_parameters=distribution.mle_parameters)
# array([1.803, 2.671])

# 100-year return value for two parameter sets
distribution.get_prop(
    prop="isf",
    x=0.01,
    free_parameters=[[-0.266, 1.353, 0.146], [-0.2, 1.35, 0.15]],
)
# array([2.670, 2.482])
```

## Custom Distributions

Any subclass of `scipy.stats.rv_continuous` can be used wherever `pyextremes`
accepts a distribution. Shape parameters are taken from the `shapes` attribute
of the distribution and are placed before `loc` and `scale`, following
the `scipy.stats` convention.

For example, a Gumbel distribution implemented from scratch produces the same
results as `gumbel_r`:

```python
import numpy as np
import scipy.stats
from pyextremes import EVA


class gumbel_gen(scipy.stats.rv_continuous):
    """Gumbel distribution for maxima."""

    def _pdf(self, x):
        return np.exp(-x - np.exp(-x))

    def _cdf(self, x):
        return np.exp(-np.exp(-x))

    def _ppf(self, q):
        return -np.log(-np.log(q))


my_gumbel = gumbel_gen(name="my_gumbel")

model = EVA(data=data)
model.get_extremes("BM", block_size="365.2425D")
model.fit_model(distribution=my_gumbel)
model.get_return_value(return_period=100)  # 2.161
```

!!! note
    `EVA` issues a `RuntimeWarning` for this distribution, because it only
    recognizes `genextreme` and `gumbel_r` as valid distributions for Block Maxima
    by name. Give custom distributions a descriptive `name`, which is shown in
    model summaries and warnings.

!!! tip
    Implementing `_cdf` and `_ppf` in addition to `_pdf` is not required,
    but `scipy.stats` otherwise calculates them numerically, which makes
    fitting and calculating return values much slower.
