# pyextremes, Extreme Value Analysis in Python
# Copyright (C), 2020 Georgii Bocharov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

if __name__ == "__main__":
    import pandas as pd
    from pyextremes import get_extremes

    data = (
        pd.read_csv("battery_wl.csv", index_col=0, parse_dates=True, squeeze=True)
        .sort_index(ascending=True)
        .dropna()
    )
    data = data.loc[data.index.year >= 1925]
    data = (
        data
        - (data.index.array - pd.to_datetime("1992")) / pd.to_timedelta("1Y") * 2.87e-3
    )
    for method in ["BM", "POT"]:
        for extremes_type in ["high", "low"]:
            if method == "BM":
                kwargs = {"block_size": "1Y", "errors": "ignore"}
            else:
                if extremes_type == "high":
                    kwargs = {"threshold": 1.35, "r": "24H"}
                else:
                    kwargs = {"threshold": -1.55, "r": "24H"}
            extremes = get_extremes(
                ts=data, method=method, extremes_type=extremes_type, **kwargs
            )
            extremes.to_csv(f"extremes_{method.lower()}_{extremes_type}.csv")
