.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

|build status| |coverage status| |pypi package|

pyextremes
==========
**pyextremes** is a Python library dedicated to solving problems from the area of `Extreme Value Analysis (EVA) <https://en.wikipedia.org/wiki/Extreme_value_theory>`_. It provides tools to extract extreme events from time series using Block Maxima or Peaks Over Threshold methods, to fit models such as GEV and GPD, and to provide estimates of extreme events and corresponding confidence intervals for given return periods. Models are fitted to the data using the Maximum Likelihood Estimate (MLE, via `scipy <https://www.scipy.org/>`_) or the Markov Chain Monte Calro (MCMC, via `pymc3 <https://docs.pymc.io/nb_tutorials/index.html>`_).

**Version:** 0.0.1

**License:** GNU General Public License v3.0

**E-Mail:** bocharovgeorgii@gmail.com

**Documentation:** work in progress

Installation
============
.. code:: bash

    pip install pyextremes

Dependencies
============
**Python version:** >=3.7

**Required packages:** numpy scipy pandas matplotlib

Tutorials
=========
Coming soon.

Illustrations
=============
Coming soon.

.. |build status| image:: https://travis-ci.org/georgebv/pyextremes.svg?branch=master
   :target: https://travis-ci.org/georgebv/pyextremes

.. |coverage status| image:: https://codecov.io/gh/georgebv/pyextremes/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/georgebv/pyextremes

.. |pypi package| image:: https://badge.fury.io/py/pyextremes.svg
    :target: https://badge.fury.io/py/pyextremes
