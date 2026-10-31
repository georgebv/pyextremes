---
title: pyextremes
---

<h1 align="center" style="font-size:50px; margin:-20px 10px -20px 10px; color:var(--md-typeset-color)">
    <em>pyextremes</em>
</h1>
<p align="center">
    <em>Extreme Value Analysis (EVA) in Python</em>
</p>
<p align="center">
<a href="https://github.com/georgebv/pyextremes/actions/workflows/test.yml" target="_blank">
    <img src="https://github.com/georgebv/pyextremes/actions/workflows/test.yml/badge.svg?event=pull_request" alt="Test">
</a>
<a href="https://codecov.io/gh/georgebv/pyextremes" target="_blank">
    <img src="https://codecov.io/gh/georgebv/pyextremes/branch/master/graph/badge.svg" alt="Coverage">
</a>
<a href="https://georgebv.github.io/pyextremes/" target="_blank">
    <img src="https://github.com/georgebv/pyextremes/actions/workflows/docs.yml/badge.svg?branch=master" alt="Docs">
</a>
<a href="https://pypi.org/project/pyextremes" target="_blank">
    <img src="https://img.shields.io/pypi/v/pyextremes.svg" alt="PyPI Package">
</a>
<a href="https://pypi.org/project/pyextremes" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/pyextremes.svg" alt="Python Versions">
</a>
<a href="https://anaconda.org/conda-forge/pyextremes" target="_blank">
    <img src="https://img.shields.io/conda/vn/conda-forge/pyextremes.svg" alt="Anaconda Package">
</a>
</p>

---

pyextremes is a Python library aimed at performing univariate
[Extreme Value Analysis (EVA)](https://en.wikipedia.org/wiki/Extreme_value_theory).

---

## Features

pyextremes provides tools necessary to perform a wide range of tasks required to
perform univariate EVA, such as:

- extraction of extreme events from time series using methods such as
  Block Maxima (BM) or Peaks Over Threshold (POT)
- fitting continuous distributions, such as GEVD, GPD, or user-specified
  continuous distributions to the extracted extreme events
- visualization of model inputs, results, and goodness-of-fit statistics
- estimation of extreme events of given probability or return period
  (e.g. 100-year event) and of corresponding confidence intervals
- tools assisting with model selection and tuning, such as selection of
  block size in BM and threshold in POT

The framework provided by pyextremes is easy to use and requires
minimal user input to get production-ready results.
Its default parameters follow established industry practice
and the underlying models are firmly grounded in Extreme Value Theory.
The largest source of inspiration for this library was the book
*An Introduction to Statistical Modeling of Extreme Values* by Stuart Coles (2001).

In addition to the easy-to-use interface, the library provides access to the
underlying tools which can be used to build custom models.
All [scipy continuous distributions](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)
are supported out-of-the-box.
Custom distributions can also be provided by subclassing
[scipy.stats.rv_continuous](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html).
Any parameter of a distribution may be frozen to investigate degenerate models
(e.g. `GEVD -> Gumbel` or `GPD -> Exponential`).

Multiple ways of fitting the continuous distributions to the data are supported:

- `MLE` (default model) - Maximum Likelihood Estimate, uses [SciPy](https://scipy.org)
- `Emcee` - Markov Chain Monte Carlo, see [Emcee](https://emcee.readthedocs.io/)
  package by Dan Foreman-Mackey
- `MOM` - Method of Moments, uses [SciPy](https://scipy.org)
- `Lmoments` - L-moments, uses [lmoments3](https://github.com/Ouranosinc/lmoments3/)

## Installation

pyextremes supports Python 3.9 to 3.14 and works with pandas 1.x, 2.x, and 3.x.

Get latest version from PyPI:

```shell
pip install pyextremes
```

Or add it to a project managed by [uv](https://docs.astral.sh/uv/):

```shell
uv add pyextremes
```

Install with all optional dependencies:

```shell
pip install "pyextremes[full]"
```

Get latest experimental build from GitHub:

```shell
pip install "pyextremes @ git+https://github.com/georgebv/pyextremes.git"
```

For Anaconda Python distributions:

```shell
conda install -c conda-forge pyextremes
```

## Dependencies

|                  Package                   | Description                                                  |
| :----------------------------------------: | :----------------------------------------------------------- |
| [**emcee**](https://emcee.readthedocs.io/) | fit models using the Markov Chain Monte Carlo method         |
| [**matplotlib**](https://matplotlib.org/)  | produce figures                                              |
|      [**numpy**](https://numpy.org/)       | perform efficient operations with arrays                     |
|  [**pandas**](https://pandas.pydata.org/)  | `Series` and `DataFrame` objects for model input and output |
|    [**scipy**](https://scipy.org/)     | statistical models and mathematical functions                 |

### Optional Dependencies

Optional dependencies can be installed all at once using the `full` extra as shown above,
individually using the `lmoments` extra (`pip install "pyextremes[lmoments]"`),
or manually:

|                 Package                                  | Description                     |
| :----------------------------------------------------:   | :------------------------------ |
| [**tqdm**](https://github.com/tqdm/tqdm)                 | progress bar for slow processes |
| [**lmoments3**](https://github.com/Ouranosinc/lmoments3) | support for Lmoments model      |

## License

This project is licensed under the terms of the
[MIT](https://opensource.org/licenses/MIT) license.
