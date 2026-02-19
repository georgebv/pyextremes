---
name: 🐞 Bug
about: Report a bug
title: '[BUG] <title>'
labels: bug
assignees: georgebv

---

<!--
Note: Please search to see if an issue already exists for the bug you encountered.
-->

### Current Behavior
<!-- A concise description of what you're experiencing. -->

### Expected Behavior
<!-- A concise description of what you expected to happen. -->

### Steps To Reproduce
<!--
Provide a code snippet and stack trace, like so:

```python
import pandas as pd

from scipy import stats

from pyextremes import EVA

# Generate random data with extreme values following GEV distribution
data = pd.Series(
    stats.genextreme.rvs(c=-0.2, loc=1.4, scale=0.15, size=100),
    index=pd.date_range(
        start="2000-01-01", periods=100, freq=pd.Timedelta("365.2425D")
    ),
)
print(data.head())

# Fit model
eva = EVA(data)
eva.get_extremes(method="BM")
eva.fit_model()
print(eva)

# Something fails
data["THIS_KEY_DOES_NOT_EXIST"]
```

```
2000-01-01 00:00:00    1.566296
2000-12-31 05:49:12    1.465512
2001-12-31 11:38:24    1.984332
2002-12-31 17:27:36    2.025562
2003-12-31 23:16:48    1.486252
Freq: 31556952s, dtype: float64
                           Univariate Extreme Value Analysis
========================================================================================
                                      Source Data
----------------------------------------------------------------------------------------
Data label:                          None      Size:                                 100
Start:                       January 2000      End:                        December 2098
========================================================================================
                                     Extreme Values
----------------------------------------------------------------------------------------
Count:                                100      Extraction method:                     BM
Type:                                high      Block size:             365 days 05:49:12
========================================================================================
                                         Model
----------------------------------------------------------------------------------------
Model:                                MLE      Distribution:                    gumbel_r
Log-likelihood:                    42.268      AIC:                              -80.412
----------------------------------------------------------------------------------------
Free parameters:                loc=1.379      Fixed parameters: All parameters are free
                              scale=0.132
========================================================================================
Traceback (most recent call last):
  File "/workspaces/pyextremes/.venv/lib/python3.13/site-packages/pandas/core/indexes/datetimes.py", line 603, in get_loc
    parsed, reso = self._parse_with_reso(key)
                   ~~~~~~~~~~~~~~~~~~~~~^^^^^
  File "/workspaces/pyextremes/.venv/lib/python3.13/site-packages/pandas/core/indexes/datetimes.py", line 559, in _parse_with_reso
    parsed, reso = super()._parse_with_reso(label)
                   ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/workspaces/pyextremes/.venv/lib/python3.13/site-packages/pandas/core/indexes/datetimelike.py", line 293, in _parse_with_reso
    parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "pandas/_libs/tslibs/parsing.pyx", line 442, in pandas._libs.tslibs.parsing.parse_datetime_string_with_reso
  File "pandas/_libs/tslibs/parsing.pyx", line 666, in pandas._libs.tslibs.parsing.dateutil_parse
pandas._libs.tslibs.parsing.DateParseError: Unknown datetime string format, unable to parse: THIS_KEY_DOES_NOT_EXIST

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspaces/pyextremes/repro.py", line 23, in <module>
    data["THIS_KEY_DOES_NOT_EXIST"]
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/pyextremes/.venv/lib/python3.13/site-packages/pandas/core/series.py", line 1130, in __getitem__
    return self._get_value(key)
           ~~~~~~~~~~~~~~~^^^^^
  File "/workspaces/pyextremes/.venv/lib/python3.13/site-packages/pandas/core/series.py", line 1246, in _get_value
    loc = self.index.get_loc(label)
  File "/workspaces/pyextremes/.venv/lib/python3.13/site-packages/pandas/core/indexes/datetimes.py", line 605, in get_loc
    raise KeyError(key) from err
KeyError: 'THIS_KEY_DOES_NOT_EXIST'
```
-->

### Environment

<!--
Example:
- OS: Ubuntu 20.04
- Python: 3.14.0
-->

### Additional Context

<!--
Add any other context about the problem here.
Can be screenshots.
-->
