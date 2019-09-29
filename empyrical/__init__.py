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
    aggregate_returns,
    alpha,
    alpha_aligned,
    alpha_beta,
    alpha_beta_aligned,
    annual_return,
    annual_volatility,
    beta,
    beta_aligned,
    cagr,
    beta_fragility_heuristic,
    beta_fragility_heuristic_aligned,
    gpd_risk_estimates,
    gpd_risk_estimates_aligned,
    calmar_ratio,
    capture,
    conditional_value_at_risk,
    cum_returns,
    cum_returns_final,
    down_alpha_beta,
    down_capture,
    downside_risk,
    excess_sharpe,
    max_drawdown,
    omega_ratio,
    roll_alpha,
    roll_alpha_aligned,
    roll_alpha_beta,
    roll_alpha_beta,
    roll_alpha_beta_aligned,
    roll_annual_volatility,
    roll_beta,
    roll_beta_aligned,
    roll_down_capture,
    roll_max_drawdown,
    roll_sharpe_ratio,
    roll_sortino_ratio,
    roll_up_capture,
    roll_up_down_capture,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
    stability_of_timeseries,
    tail_ratio,
    up_alpha_beta,
    up_capture,
    up_down_capture,
    value_at_risk,
)

from .periods import (
    DAILY,
    WEEKLY,
    MONTHLY,
    QUARTERLY,
    YEARLY
)


from .perf_attrib import (
    perf_attrib,
    compute_exposures,
)
