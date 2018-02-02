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
# flake8: noqa

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .stats import (
    cum_returns,
    cum_returns_final,
    aggregate_returns,
    max_drawdown,
    annual_return,
    annual_volatility,
    calmar_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    downside_risk,
    excess_sharpe,
    alpha_beta,
    alpha,
    beta,
    alpha_beta_aligned,
    alpha_aligned,
    beta_aligned,
    stability_of_timeseries,
    tail_ratio,
    cagr,
    capture,
    up_capture,
    down_capture,
    up_down_capture,
    up_alpha_beta,
    down_alpha_beta,
    roll_max_drawdown,
    roll_up_capture,
    roll_down_capture,
    roll_up_down_capture,
    roll_alpha_beta,
    roll_sharpe_ratio,
    value_at_risk,
    conditional_value_at_risk,
)

from .periods import (
    DAILY,
    WEEKLY,
    MONTHLY,
    YEARLY
)
