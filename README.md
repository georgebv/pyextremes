<p align="center" style="font-size:40px; margin:0px 10px 0px 10px">
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

# About

**Documentation:** https://georgebv.github.io/pyextremes/

**License:** [MIT](https://opensource.org/licenses/MIT)

**Support:** [ask a question](https://github.com/georgebv/pyextremes/discussions)
or [create an issue](https://github.com/georgebv/pyextremes/issues/new/choose),
any input is appreciated and would help develop the project

**pyextremes** is a Python library aimed at performing univariate
[Extreme Value Analysis (EVA)](https://en.wikipedia.org/wiki/Extreme_value_theory).
It provides tools necessary to perform a wide range of tasks required to
perform EVA, such as:

- extraction of extreme events from time series using methods such as
  Block Maxima (BM) or Peaks Over Threshold (POT)
- fitting continuous distributions, such as GEVD, GPD, or user-specified
  continuous distributions to the extracted extreme events
- visualization of model inputs, results, and goodness-of-fit statistics
- estimation of extreme events of given probability or return period
  (e.g. 100-year event) and of corresponding confidence intervals
- tools assisting with model selection and tuning, such as selection of
  block size in BM and threshold in POT

Multiple ways of fitting distributions to the extreme events are supported:

- `MLE` (default) - Maximum Likelihood Estimate, uses [SciPy](https://scipy.org)
- `Emcee` - Markov Chain Monte Carlo, uses [emcee](https://emcee.readthedocs.io/)
- `MOM` - Method of Moments, uses [SciPy](https://scipy.org)
- `Lmoments` - L-moments, uses [lmoments3](https://github.com/Ouranosinc/lmoments3)

# Installation

pyextremes supports Python 3.9 to 3.14 and works with pandas 1.x, 2.x, and 3.x.

Get latest version from PyPI:

```shell
pip install pyextremes
```

Or add it to a project managed by [uv](https://docs.astral.sh/uv/):

```shell
uv add pyextremes
```

Install with all optional dependencies (`tqdm` progress bars and the `Lmoments` model):

```shell
pip install "pyextremes[full]"
```

Get latest experimental build from GitHub:

```shell
pip install "pyextremes @ git+https://github.com/georgebv/pyextremes.git"
```

Get pyextremes for the Anaconda Python distribution:

```shell
conda install -c conda-forge pyextremes
```

# Quick Start

```python
import pandas as pd
from pyextremes import EVA

data = pd.read_csv("battery_wl.csv", index_col=0, parse_dates=True).squeeze()

model = EVA(data)
model.get_extremes(method="BM", block_size="365.2425D")
model.fit_model()

summary = model.get_summary(
    return_period=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
    alpha=0.95,
    n_samples=1000,
)
model.plot_diagnostic(alpha=0.95)
```

When running as a script, put this code under `if __name__ == "__main__":`,
because confidence intervals are calculated in parallel using `multiprocessing`.

See the [Quick Start](https://georgebv.github.io/pyextremes/quickstart/) for the full example
(including the data and its preparation) and the
[User Guide](https://georgebv.github.io/pyextremes/user-guide/1-read-first/) for in-depth tutorials.

# Illustrations

<p align="center" style="font-size:20px; margin:10px 10px 0px 10px">
    <em>Model diagnostic</em>
</p>
<p align="center" style="font-size:20px; margin:10px 10px 40px 10px">
  <img src="https://raw.githubusercontent.com/georgebv/pyextremes/master/docs/img/quickstart/diagnostic.png" alt="Diagnostic plot" width="600px">
</p>

<p align="center" style="font-size:20px; margin:10px 10px 0px 10px">
    <em>Extreme value extraction</em>
</p>
<p align="center" style="font-size:20px; margin:10px 10px 40px 10px">
  <img src="https://raw.githubusercontent.com/georgebv/pyextremes-notebooks/master/notebooks/documentation/readme%20figures/extremes.png" alt="Extreme value extraction" width="600px">
</p>

<p align="center" style="font-size:20px; margin:10px 10px 0px 10px">
    <em>Trace plot</em>
</p>
<p align="center" style="font-size:20px; margin:10px 10px 40px 10px">
  <img src="https://raw.githubusercontent.com/georgebv/pyextremes/master/docs/img/models/mcmc-trace.png" alt="Trace plot" width="600px">
</p>

<p align="center" style="font-size:20px; margin:10px 10px 0px 10px">
    <em>Corner plot</em>
</p>
<p align="center" style="font-size:20px; margin:10px 10px 40px 10px">
  <img src="https://raw.githubusercontent.com/georgebv/pyextremes/master/docs/img/models/mcmc-corner.png" alt="Corner plot" width="600px">
</p>

# Acknowledgements

I wanted to thank Max Larson who has inspired me to start this project
and who taught me a lot about extreme value theory.
