.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

|build status| |coverage status| |pypi package| |conda version|

pyextremes
==========
**pyextremes** is a Python library dedicated to solving problems from the area of `Extreme Value Analysis (EVA) <https://en.wikipedia.org/wiki/Extreme_value_theory>`_. It provides tools to extract extreme events from time series using Block Maxima or Peaks Over Threshold methods, to fit models such as GEV and GPD to the extracted extreme values, and to provide estimates of extreme events and corresponding confidence intervals for given return periods. Models are fitted to the data using the Maximum Likelihood Estimate (MLE, via `scipy <https://www.scipy.org/>`_) or the Markov Chain Monte Calro (MCMC, via `emcee <https://emcee.readthedocs.io/en/stable/>`_) models.

**Version:** 1.1.0

**License:** GNU General Public License v3.0

**E-Mail:** bocharovgeorgii@gmail.com

**Documentation:** coming soon

Installation
============
Available via pip:

.. code:: bash

    pip install pyextremes

And via anaconda:

.. code:: bash

    conda install -c conda-forge pyextremes

Dependencies
============
**Python version:** 3.7 or later

**Required packages:**

- emcee >= 2.2.1
- matplotlib >= 3.1.3
- numpy >= 1.18.1
- pandas >= 1.0.1
- scipy >= 1.4.1

Tutorials
=========
- `Basic usage <https://nbviewer.jupyter.org/github/georgebv/pyextremes-notebooks/blob/master/notebooks/EVA%20basic.ipynb>`_
- Models
- Statistical distributions

Illustrations
=============
Extreme value extraction

|extreme events image|

Model diagnostic

|model diagnostic image|

.. |build status| image:: https://travis-ci.org/georgebv/pyextremes.svg?branch=master
   :target: https://travis-ci.org/georgebv/pyextremes

.. |coverage status| image:: https://codecov.io/gh/georgebv/pyextremes/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/georgebv/pyextremes

.. |pypi package| image:: https://badge.fury.io/py/pyextremes.svg
    :target: https://pypi.org/project/pyextremes/

.. |conda version| image:: https://img.shields.io/conda/vn/conda-forge/pyextremes.svg
    :target: https://anaconda.org/conda-forge/pyextremes

.. |extreme events image| image:: ./docs/source/example_images/extremes.png

.. |model diagnostic image| image:: ./docs/source/example_images/diagnostic.png
