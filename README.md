[![Build Status](https://travis-ci.org/quantopian/empyrical.svg?branch=master)](https://travis-ci.org/quantopian/empyrical)

![PyPI](https://img.shields.io/pypi/v/empyrical?color=%234ec726&style=flat-square)

# empyrical

Common financial risk metrics.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)
- [Testing](#testing)

## Installation
```
pip install empyrical
```

## Usage

Simple Statistics
```python
import numpy as np
from empyrical import max_drawdown, alpha_beta

returns = np.array([.01, .02, .03, -.4, -.06, -.02])
benchmark_returns = np.array([.02, .02, .03, -.35, -.05, -.01])

# calculate the max drawdown
max_drawdown(returns)

# calculate alpha and beta
alpha, beta = alpha_beta(returns, benchmark_returns)

```

Rolling Measures
```python
import numpy as np
from empyrical import roll_max_drawdown

returns = np.array([.01, .02, .03, -.4, -.06, -.02])

# calculate the rolling max drawdown
roll_max_drawdown(returns, window=3)

```

Pandas Support
```python
import pandas as pd
from empyrical import roll_up_capture, capture

returns = pd.Series([.01, .02, .03, -.4, -.06, -.02])

# calculate a capture ratio
capture(returns)

# calculate capture for up markets on a rolling 60 day basis
roll_up_capture(returns, window=60)
```

## Support

Please [open an issue](https://github.com/quantopian/empyrical/issues/new) for support.

### Deprecated: Data Reading via `pandas-datareader`

As of early 2018, Yahoo Finance has suffered major API breaks with no stable
replacement, and the Google Finance API has not been stable since late 2017
[(source)](https://github.com/pydata/pandas-datareader/blob/da18fbd7621d473828d7fa81dfa5e0f9516b6793/README.rst).
In recent months it has become a greater and greater strain on the `empyrical`
development team to maintain support for fetching data through
`pandas-datareader` and other third-party libraries, as these APIs are known to
be unstable.

As a result, all `empyrical` support for data reading functionality has been
deprecated and will be removed in a future version.

Users should beware that the following functions are now deprecated:

- `empyrical.utils.cache_dir`
- `empyrical.utils.data_path`
- `empyrical.utils.ensure_directory`
- `empyrical.utils.get_fama_french`
- `empyrical.utils.load_portfolio_risk_factors`
- `empyrical.utils.default_returns_func`
- `empyrical.utils.get_symbol_returns_from_yahoo`

Users should expect regular failures from the following functions, pending
patches to the Yahoo or Google Finance API:

- `empyrical.utils.default_returns_func`
- `empyrical.utils.get_symbol_returns_from_yahoo`

## Contributing

Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/quantopian/empyrical/compare/).

## Testing
- install requirements
  - "nose>=1.3.7",
  - "parameterized>=0.6.1"

```
./runtests.py
```
