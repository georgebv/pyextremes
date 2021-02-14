<p align="center" style="font-size:40px; margin:0px 10px 0px 10px">
    <em>pyextremes</em>
</p>
<p align="center">
    <em>Extreme Value Analysis (EVA) in Python</em>
</p>
<p align="center">
<a href="https://github.com/georgebv/pyextremes/actions?query=workflow%3Abuild" target="_blank">
    <img src="https://github.com/georgebv/pyextremes/workflows/build/badge.svg" alt="Test">
</a>
<a href="https://codecov.io/gh/georgebv/pyextremes" target="_blank">
    <img src="https://codecov.io/gh/georgebv/pyextremes/branch/master/graph/badge.svg" alt="Coverage">
</a>
<a href="https://pypi.org/project/pyextremes" target="_blank">
    <img src="https://badge.fury.io/py/pyextremes.svg" alt="PyPI Package">
</a>
<a href="https://anaconda.org/conda-forge/pyextremes" target="_blank">
    <img src="https://img.shields.io/conda/vn/conda-forge/pyextremes.svg" alt="Anaconda Package">
</a>
</p>

---

**pyextremes** is a Python library aimed at performing univariate and multivariate
[Extreme Value Analysis (EVA)](https://en.wikipedia.org/wiki/Extreme_value_theory).

---

## Features

**pyextremes** provides tools necessary to perform a wide range of tasks required to
perform EVA, such as:

- extraction of extreme events from time series using methods such as
Block Maxima (BM) or Peaks Over Threshold (POT)
- fitting continuous distributions, such as GEVD, GPD, or user-specified
continous distributions to the extracted extreme events
- visualization of model inputs, results, and goodness-of-fit statistics
- estimation of extreme events of given probability or return period
(e.g. 100-year event) and of corresponding confidence intervals
- tools assisting with model selection and tuning, such as selection of
block size in BM and threshold in POT

Framework provided by the **pyextremes** library is easy to use and requires
minimum user input to get production-ready results.
Its default parameters are configured in compliance with best industry standards
and underlying models are heavily based in the Extreme Value theory.
The largest source of inspiration for this library was the book
`"An Introduction to Statistical Modeling of Extreme Values"` by Stuart Coles.

In addition to the easy-to-use interface, the library provides interface to
underlying tools which can be used to build custom models.
All [scipy continuous distributions](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)
are supported out-of-the-box.
Custom distributions can be also provided by subclassing
[scipy.stats.rv_continuous](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html).
Any parameter of a distribution may be frozen to investigate degenerate models
(e.g. `GEVD -> Gumbel` or `GPD -> Exponential`).

Multiple ways of fitting the continuous distributions to the data are supported:

- `MLE` (default model) - Maximum Likelihood Estimate, uses [SciPy](https://www.scipy.org)
- `Emcee` - Markov Chain Monte Calro, see [Emcee](https://emcee.readthedocs.io/en/stable/)
package by Dan Foreman-Mackey

## Installation

### pip

Get latest version from PyPI:

```shell
pip install pyextremes
```

Get latest experimental build from GitHub:

```shell
pip install git+https://github.com/georgebv/pyextremes
```

### Anaconda

```shell
conda install -c conda-forge pyextremes
```

## Dependencies

|                  Package                   | Description                                                  |
| :----------------------------------------: | :----------------------------------------------------------- |
| [**emcee**](https://emcee.readthedocs.io/) | fit models using the Markov Chain Monte Carlo method         |
| [**matplotlib**](https://matplotlib.org/)  | produce figures                                              |
|      [**numpy**](https://numpy.org/)       | perform efficient operations with arrays                     |
|  [**pandas**](https://pandas.pydata.org/)  | `Series` and `DataFrame` objects for model intput and output |
|    [**scipy**](https://www.scipy.org/)     | statistical models and mathematic functions                  |


## Example

### Read data

Every **pyextremes** model starts with a `pandas.Series`
(see [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)) object,
which contains timeseries of data you want to analyze.
This example is based on water level data for "The Battery" station located in New York.

Read data:

```python linenums="1"
import pandas as pd

series = pd.read_csv(
    "battery_wl.csv",
    index_col=0,
    parse_dates=True,
    squeeze=True,
)
```

???+ tip
    The `battery_wl.csv` file referenced above is used throughout many tutorials
    and examples for the `pyextremes` package.
    If you want to reproduce all steps shown here and get the same results,
    the file can be downloaded
    [here](https://github.com/georgebv/pyextremes-notebooks/tree/master/data).

---

### Clean up data

In order for the analysis results to be meaningful, data needs to be pre-processed
by the user. This may include removal of data gaps, detrending, interpolation,
removal of outliers, etc.
Let's clean up the data:

```python linenums="9"
series = (
    series
    .sort_index(ascending=True)
    .astype(float)
    .dropna()
)
series = series.loc[pd.to_datetime("1925"):]
series = series - (series.index.array - pd.to_datetime("1992")) / pd.to_timedelta("365.2425D") * 2.87e-3
```

??? note 
    See [this tutorial](https://nbviewer.jupyter.org/github/georgebv/pyextremes-notebooks/blob/master/notebooks/EVA%20basic.ipynb) for more information on why these specific operations were done.


This is how `series` looks like in Python terminal:

```python
print(series.head())
```

Result:

```
Date-Time (GMT)
1926-11-20 05:00:00   -0.411120
1926-11-20 06:00:00   -0.777120
1926-11-20 07:00:00   -1.051120
1926-11-20 08:00:00   -1.051121
1926-11-20 09:00:00   -0.808121
Name: Water Elevation [m NAVD88], dtype: float64
```

---

### Create model

The primary interface to the pyextremes library is provided via the `EVA` class.
This class is responsible for all major tasks outlined above and is created using
a simple command:

```python linenums="17"
from pyextremes import EVA

model = EVA(series)
```

---

### Extract extreme values

The first step of extreme value analysis is extraction of extreme values from the
timeseries. This is done by using the `get_extremes` method of the `EVA` class.

In this example extremes will be extracted using the BM `method` and 1-year
`block_size`, which give us annual maxima series.

```python linenums="20"
model.get_extremes(method="BM", block_size="365.2425D")
```

View timeseries of the extracted extremes:

```python
print(model.extremes.head())
```

```
Date-Time (GMT)
1927-02-20 16:00:00    1.670154
1927-12-05 10:00:00    1.432893
1929-04-16 19:00:00    1.409977
1930-08-23 01:00:00    1.202101
1931-03-08 17:00:00    1.529547
Name: Water Elevation [m NAVD88], dtype: float64
```

Visualize the extremes:

```python
model.plot_extremes()
```

![BM extremes](https://raw.githubusercontent.com/georgebv/pyextremes-notebooks/master/notebooks/documentation/documentation%20figures/index%20extremes.png){: align=center }

---

### Fit a model

The next step is selecting a model and fitting to the extracted extreme events.
What this means practically is that we need to find model parameters
(such as shape, location and scale for GEVD or GPD)
that maximize or minimize some metric (likelihood) and give us the best fit possible.
This is done by calling the `fit_model` method:

```python linenums="21"
model.fit_model()
```

!!! info "Info"
    By default, the `fit_model` method selects the best model applicable
    to extracted extremes using the Akaike Information Criterion (AIC).

---

### Calculate return values

The final goal of most EVA's is estimation of return values.
The simplest way to do this is by using the `get_summary` method:

```python linenums="22"
summary = model.get_summary(
    return_period=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
    alpha=0.95,
    n_samples=1000,
)
```

???+ note
    By default return period size is set to one year,
    which is defined as the mean year from the Gregorian calendar
    ([365.2425 days](https://en.wikipedia.org/wiki/Year)).
    This means that a return period of 100 corresponds to a 100-year event.
    
    A different return period size can be specified using the `return_period_size`
    argument. A value of `30D` (30 days) would mean that a return period of 12
    corresponds to approximately one year.

Print the results:

```python
print(summary)
```

```
               return value  lower ci  upper ci
return period                                  
1.0                0.802610 -0.270608  1.024385
2.0                1.409343  1.370929  1.452727
5.0                1.622565  1.540408  1.710116
10.0               1.803499  1.678816  1.955386
25.0               2.090267  1.851597  2.417670
50.0               2.354889  1.992022  2.906734
100.0              2.671313  2.145480  3.568418
250.0              3.188356  2.346609  4.856107
500.0              3.671580  2.517831  6.232830
1000.0             4.252220  2.702800  8.036243
```

---

### Investigate model

After model results are obtained, logical questions naturally arise -
how good is the model, are the obtained results meaningful, and
how confident can I be with the estimated return values.
One way to do that is by visually inspecting the model:

```python linenums="27"
model.plot_diagnostic(alpha=0.95)
```

![Diagnostic plot](https://raw.githubusercontent.com/georgebv/pyextremes-notebooks/master/notebooks/documentation/documentation%20figures/index%20diagnostic.png){: align=center }

---

### Recap

Following this example you should be able to do the following:

- set up an `EVA` instance
- extract extreme events
- fit a model
- get results

For more in-depth tutorials on features of pyextremes see the rest of documentation.

---

## License

This project is licensed under the terms of the
[MIT](https://opensource.org/licenses/MIT) license.

--8<-- "docs/includes/abbreviations.md"
