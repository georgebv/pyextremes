<p align="center" style="font-size:50px; margin:-20px 10px -20px 10px">
    <em>pyextremes</em>
</p>
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
<a href="https://pypi.org/project/pyextremes" target="_blank">
    <img src="https://badge.fury.io/py/pyextremes.svg" alt="PyPI Package">
</a>
<a href="https://anaconda.org/conda-forge/pyextremes" target="_blank">
    <img src="https://img.shields.io/conda/vn/conda-forge/pyextremes.svg" alt="Anaconda Package">
</a>
</p>

---

pyextremes is a Python library aimed at performing univariate
[Extreme Value Analysis (EVA)](https://en.wikipedia.org/wiki/Extreme_value_theory)

---

## Features

pyextremes provides tools necessary to perform a wide range of tasks required to
perform univariate EVA, such as:

- extraction of extreme events from time series using methods such as
  Block Maxima (BM) or Peaks Over Threshold (POT)
- fitting continuous distributions, such as GEVD, GPD, or user-specified
  continous distributions to the extracted extreme events
- visualization of model inputs, results, and goodness-of-fit statistics
- estimation of extreme events of given probability or return period
  (e.g. 100-year event) and of corresponding confidence intervals
- tools assisting with model selection and tuning, such as selection of
  block size in BM and threshold in POT

Framework provided by the pyextremes library is easy to use and requires
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
- `MOM` - Method of Moments, uses SciPy
- `Lmoments` - L-moment, see [lmoments3][https://github.com/Ouranosinc/lmoments3/] package

## Installation

Get latest version from PyPI:

```shell
pip install pyextremes
```

Install with optional dependencies:

```shell
pip install pyextremes[full]
```

Get latest experimental build from GitHub:

```shell
pip install "git+https://github.com/georgebv/pyextremes.git#egg=pyextremes"
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
|  [**pandas**](https://pandas.pydata.org/)  | `Series` and `DataFrame` objects for model intput and output |
|    [**scipy**](https://www.scipy.org/)     | statistical models and mathematic functions                  |

### Optional Dependencies

Optional dependencies can be installed using the `full` tag as shown above or manually:

|                 Package                                | Description                     |
| :----------------------------------------------------: | :------------------------------ |
| [**tqdm**](https://github.com/tqdm/tqdm)               | progress bar for slow processes |
| [**lmoments3](https://github.com/Ouranosinc/lmoments3) | support for Lmoments model      |

## License

This project is licensed under the terms of the
[MIT](https://opensource.org/licenses/MIT) license.

--8<-- "docs/includes/abbreviations.md"
