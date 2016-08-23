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

from __future__ import division

import pandas as pd
import numpy as np
from scipy import stats

from empyrical.utils import nanmean


APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    YEARLY: 1
}


def annualization_factor(period, annualization):
    """
    Return annualization factor from period entered or if a custom
    value is passed in.

    Parameters
    ----------
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Annualization factor.
    """
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def cum_returns(returns, starting_value=0):
    """
    Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns : pd.Series
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902.
    starting_value : float, optional
       The starting returns.

    Returns
    -------
    pandas.Series
        Series of cumulative returns.

    Notes
    -----
    For increased numerical accuracy, convert input to log returns
    where it is possible to sum instead of multiplying.
    PI((1+r_i)) - 1 = exp(ln(PI(1+r_i)))     # x = exp(ln(x))
                    = exp(SIGMA(ln(1+r_i))   # ln(a*b) = ln(a) + ln(b)
    """
    # df_price.pct_change() adds a nan in first position, we can use
    # that to have cum_logarithmic_returns start at the origin so that
    # df_cum.iloc[0] == starting_value
    # Note that we can't add that ourselves as we don't know which dt
    # to use.

    if len(returns) < 1:
        return np.nan

    if pd.isnull(returns.iloc[0]):
        returns.iloc[0] = 0.

    df_cum = np.exp(np.log1p(returns).cumsum())

    if starting_value == 0:
        return df_cum - 1
    else:
        return df_cum * starting_value


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    returns : pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    pd.Series
        Aggregated returns.
    """

    def cumulate_returns(x):
        return cum_returns(x)[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )

    return returns.groupby(grouping).apply(cumulate_returns)


def max_drawdown(returns):
    """
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    float
        Maximum drawdown.

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """

    if len(returns) < 1:
        return np.nan

    cumulative = cum_returns(returns, starting_value=100)
    max_return = cumulative.cummax()
    return cumulative.sub(max_return).div(max_return).min()


def annual_return(returns, period=DAILY, annualization=None):
    """Determines the mean annual growth rate of returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Annual Return as CAGR (Compounded Annual Growth Rate).

    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    num_years = float(len(returns)) / ann_factor
    start_value = 100
    end_value = cum_returns(returns, starting_value=start_value).iloc[-1]
    total_return = (end_value - start_value) / start_value
    annual_return = (1. + total_return) ** (1. / num_years) - 1

    return annual_return


def annual_volatility(returns, period=DAILY, alpha=2.0,
                      annualization=None):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    alpha : float, optional
        Scaling relation (Levy stability exponent).
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Annual volatility.
    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    volatility = returns.std() * (ann_factor ** (1.0 / alpha))

    return volatility


def calmar_ratio(returns, period=DAILY, annualization=None):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.


    Returns
    -------
    float
        Calmar ratio (drawdown ratio) as float. Returns np.nan if there is no
        calmar ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    max_dd = max_drawdown(returns=returns)
    if max_dd < 0:
        temp = annual_return(
            returns=returns,
            period=period,
            annualization=annualization
        ) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def omega_ratio(returns, risk_free=0.0, required_return=0.0,
                annualization=APPROX_BDAYS_PER_YEAR):
    """Determines the Omega ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period
    required_return : float, optional
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g. An annual minimum
        acceptable return of 100 will translate to a minimum acceptable
        return of 0.018.
    annualization : int, optional
        Factor used to convert the required_return into a daily
        value. Enter 1 if no time period conversion is necessary.

    Returns
    -------
    float
        Omega ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.

    """

    if len(returns) < 2:
        return np.nan

    if annualization == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** \
            (1. / annualization) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sharpe_ratio(returns, risk_free=0, period=DAILY, annualization=None):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Sharpe ratio.
    np.nan
        If insufficient length of returns or if if adjusted returns are 0.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.

    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    returns_risk_adj = returns - risk_free

    if np.std(returns_risk_adj, ddof=1) == 0:
        return np.nan

    return np.mean(returns_risk_adj) / np.std(returns_risk_adj, ddof=1) * \
        np.sqrt(ann_factor)


def sortino_ratio(returns, required_return=0, period=DAILY,
                  annualization=None):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> pd.Series

        Annualized Sortino ratio.

    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    if len(returns) < 2:
        return np.nan

    mu = nanmean(returns - required_return, axis=0)
    sortino = mu / downside_risk(returns, required_return)
    if len(returns.shape) == 2:
        sortino = pd.Series(sortino, index=returns.columns)
    return sortino * ann_factor


def downside_risk(returns, required_return=0, period=DAILY,
                  annualization=None):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> pd.Series

        Annualized downside deviation

    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    downside_diff = returns - required_return
    mask = downside_diff > 0
    downside_diff[mask] = 0.0
    squares = np.square(downside_diff)
    mean_squares = nanmean(squares, axis=0)
    dside_risk = np.sqrt(mean_squares) * np.sqrt(ann_factor)
    if len(returns.shape) == 2:
        dside_risk = pd.Series(dside_risk, index=returns.columns)
    return dside_risk


def information_ratio(returns, factor_returns):
    """
    Determines the Information ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns: float / series
        Benchmark return to compare returns against.

    Returns
    -------
    float
        The information ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/information_ratio for more details.

    """
    if len(returns) < 2:
        return np.nan

    active_return = returns - factor_returns
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.mean(active_return) / tracking_error


def alpha_beta(returns, factor_returns, risk_free=0.0, period=DAILY,
               annualization=None):
    """Calculates annualized alpha and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Alpha.
    float
        Beta.

    """
    b = beta(returns, factor_returns, risk_free)
    a = alpha(returns, factor_returns, risk_free, period, annualization)
    return a, b


def alpha(returns, factor_returns, risk_free=0.0, period=DAILY,
          annualization=None):
    """Calculates annualized alpha.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~empyrical.stats.annual_return`.

    Returns
    -------
    float
        Alpha.
    """
    if len(returns) < 2:
        return np.nan
    ann_factor = annualization_factor(period, annualization)
    b = beta(returns, factor_returns, risk_free)
    alpha = returns - risk_free - b*(factor_returns - risk_free)
    return alpha.mean() * ann_factor


def beta(returns, factor_returns, risk_free=0.0):
    """Calculates beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    Returns
    -------
    float
        Beta.
    """

    # Filter out dates with np.nan as a return value
    indices = np.intersect1d(
        returns.dropna().index,
        factor_returns.dropna().index
    )

    if len(indices) < 2:
        return np.nan

    covar = np.cov(returns[indices]-risk_free,
                   factor_returns[indices], ddof=0)[0][1]

    return covar/np.var(factor_returns[indices])


def stability_of_timeseries(returns):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    float
        R-squared.

    """
    if len(returns) < 2:
        return np.nan

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)),
                            cum_log_returns.values)[2]

    return rhat ** 2


def tail_ratio(returns):
    """Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    float
        tail ratio

    """

    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))


def cagr(returns, period=DAILY, annualization=None):
    """
    Compute compound annual growth rate.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~empyrical.stats.annual_return`.

    Returns
    -------
    float, np.nan
        The CAGR value.

    """
    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    no_years = len(returns) / float(ann_factor)
    ending_value = cum_returns(returns, starting_value=1).iloc[-1]

    return ending_value ** (1. / no_years) - 1


def sqn(trade_returns):
    """Determines the SystemQualityNumber of trades as defined by Van K. Tharp.

    Parameters
    ----------
    trade_returns : pd.Series
        A series of transaction gains/losses.

    Returns
    -------
    float
        SystemQualityNumber

    """
    n_trades = len(trade_returns.values)

    if n_trades < 30:
        return np.nan

    pos = trade_returns[trade_returns > 0]

    if len(pos) < 3:
        return np.nan

    std = np.nanstd(pos, ddof=1)
    avg = np.nanmean(pos)

    if std == 0:
        return np.nan

    return np.sqrt(n_trades) * avg / std


SIMPLE_STAT_FUNCS = [
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    stats.skew,
    stats.kurtosis,
    tail_ratio,
    cagr
]

FACTOR_STAT_FUNCS = [
    information_ratio,
    alpha,
    beta,
]

TRADE_STAT_FUNCS = [
    sqn,
]
