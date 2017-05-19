#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pandas as pd

try:
    # fast versions
    import bottleneck as bn
    nanmean = bn.nanmean
    nanstd = bn.nanstd
    nansum = bn.nansum
    nanmax = bn.nanmax
    nanmin = bn.nanmin
    nanargmax = bn.nanargmax
    nanargmin = bn.nanargmin
except ImportError:
    nanmean = np.nanmean
    nanstd = np.nanstd
    nansum = np.nansum
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin

def roll(*args, **kwargs):
    func, kwargs = _pop_kwargs('functions', kwargs)
    window = kwargs.pop('window')
    if len(args) > 2:
        raise ValueError("Cannot pass more than 2 return sets")

    if len(args) == 2:
        if type(arg[0]) != type(arg[0]):
            raise ValueError("The two returns arguments are not the same.")

    if isinstance(args[0], np.ndarray):
        return _roll_ndarray(func, window, *args, **kwargs)
    return _roll_pandas(func, window, *args, **kwargs)

def up(returns, factor_returns, **kwargs):
    func, kwargs = _pop_kwargs('functions', kwargs)
    returns = returns[factor_returns > 0]
    factor_returns = factor_returns[factor_returns > 0]
    return func(returns, factor_returns, **kwargs)

def down(returns, factor_returns, **kwargs):
    func, kwargs = _pop_kwargs('functions', kwargs)
    returns = returns[factor_returns < 0]
    factor_returns = factor_returns[factor_returns < 0]
    return func(returns, factor_returns, **kwargs)

def _pop_kwargs(sym, kwargs):
    funcs = kwargs.pop(sym)
    func = funcs[0]
    if funcs[1:]:
        kwargs[sym] = funcs[1:]
    return func, kwargs

def _roll_ndarray(func, window, *args, **kwargs):
    data = []
    for i in range(window, len(args[0])):
        rets = [s[i-window:i] for s in args]
        data.append(func(*rets, **kwargs))
    return np.array(data)

def _roll_pandas(func, window, *args, **kwargs):
    data = {}
    for i in range(window, len(args[0])):
        rets = [s.iloc[i-window:i] for s in args]
        data[args[0].index[i]] = func(*rets, **kwargs)
    return pd.Series(data)
