<p align="center">
  <a href="https://github.com/georgebv/pyextremes"><img src="https://raw.githubusercontent.com/georgebv/pyextremes-notebooks/master/notebooks/documentation/readme%20figures/diagnostic.png" alt="pyextremes" width="50%"></a>
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

**pyextremes** is a Python library aimed at performing univariate and multivariate (work-in-progress)
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

*[EVA]: Extreme Value Analysis
*[BM]: Block Maxima
*[POT]: Peaks over Threshold
*[GEVD]: Generalized Extreme Value Distribution
*[GPD]: Generalized Pareto Distribution

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

### Prepare data

Every **pyextremes** model starts with a `pandas.Series` object,
which contains timeseries of a parameter which extremes are to be analyzed.

```python linenums="1"
import pandas as pd

series = pd.read_csv("rain data.csv", parse_dates=True, squeeze=True)
```

### Create model

The primary interface to the pyextremes library is provided via the `EVA` class.
This class is responsible for all major tasks outlined above and is created using
as simple command:

```python linenums="4"
from pyextremes import EVA

model = EVA(series)
```

### Extract extreme values

The first step of extreme value analysis is extraction of extreme values from the
timeseries. This is done by using the `get_extremes` method of the `EVA` class:

```python linenums="7"
model.get_extremes(
    method="BM",
    extremes_type="high",
    block_size="365.2425D",
    errors="raise",
)
```

### Fit a model

```python linenums="13"
model.fit_model()
```

!!! info "Info"
    By default, the `fit_model` method selects the best model applicable
    to extracted extremes using the Akaike Information Criterion (AIC).

### Calculate return values

```python linenums="14"
summary = model.get_summary(
    return_period=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
    alpha=0.95,
    n_samples=1000,
)
print(summary)
```

## License

This project is licensed under the [MIT](https://opensource.org/licenses/MIT) license.
