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
from math import pow
from scipy import stats, optimize
from six import iteritems
from sys import float_info

from .utils import nanmean, nanstd, nanmin


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


def _adjust_returns(returns, adjustment_factor):
    """
    Returns the returns series adjusted by adjustment_factor. Optimizes for the
    case of adjustment_factor being 0 by returning returns itself, not a copy!

    Parameters
    ----------
    returns : pd.Series or np.ndarray
    adjustment_factor : pd.Series or np.ndarray or float or int

    Returns
    -------
    pd.Series or np.ndarray
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns
    return returns - adjustment_factor


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
    returns : pd.Series, np.ndarray, or pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902.
        - Also accepts two dimensional data. In this case,
            each column is cumulated.
    starting_value : float, optional
       The starting returns.

    Returns
    -------
    pd.Series, np.ndarray, or pd.DataFrame
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
        return type(returns)([])

    if np.any(np.isnan(returns)):
        returns = returns.copy()
        returns[np.isnan(returns)] = 0.

    df_cum = (returns + 1).cumprod(axis=0)

    if starting_value == 0:
        return df_cum - 1
    else:
        return df_cum * starting_value


def cum_returns_final(returns, starting_value=0):
    """
    Compute total returns from simple returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
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
    float

    """

    if len(returns) == 0:
        return np.nan

    return cum_returns(np.asanyarray(returns),
                       starting_value=starting_value)[-1]


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
        return cum_returns(x).iloc[-1]

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
    returns : pd.Series or np.ndarray
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
    max_return = np.fmax.accumulate(cumulative)
    return nanmin((cumulative - max_return) / max_return)


def annual_return(returns, period=DAILY, annualization=None):
    """Determines the mean annual growth rate of returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
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
    # Pass array to ensure index -1 looks up successfully.
    end_value = cum_returns(np.asanyarray(returns),
                            starting_value=start_value)[-1]
    cum_returns_final = (end_value - start_value) / start_value
    annual_return = (1. + cum_returns_final) ** (1. / num_years) - 1

    return annual_return


def annual_volatility(returns, period=DAILY, alpha=2.0,
                      annualization=None):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
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

    volatility = nanstd(returns, ddof=1) * (ann_factor ** (1.0 / alpha))

    return volatility


def calmar_ratio(returns, period=DAILY, annualization=None):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
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
    returns : pd.Series or np.ndarray
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
    returns : pd.Series or np.ndarray
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

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    returns_risk_adj = returns_risk_adj[~np.isnan(returns_risk_adj)]

    if np.std(returns_risk_adj, ddof=1) == 0:
        return np.nan

    return np.mean(returns_risk_adj) / np.std(returns_risk_adj, ddof=1) * \
        np.sqrt(ann_factor)


def sortino_ratio(returns, required_return=0, period=DAILY,
                  annualization=None, _downside_risk=None):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
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
    _downside_risk : float, optional
        The downside risk of the given inputs, if known. Will be calculated if
        not provided.

    Returns
    -------
    float, pd.Series

        depends on input type
        series ==> float
        DataFrame ==> pd.Series

        Annualized Sortino ratio.

    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    adj_returns = _adjust_returns(returns, required_return)
    mu = nanmean(adj_returns, axis=0)
    dsr = (_downside_risk if _downside_risk is not None
           else downside_risk(returns, required_return))
    sortino = mu / dsr
    return sortino * ann_factor


def downside_risk(returns, required_return=0, period=DAILY,
                  annualization=None):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
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
    float, pd.Series
        depends on input type
        series ==> float
        DataFrame ==> pd.Series

        Annualized downside deviation

    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    downside_diff = _adjust_returns(returns, required_return).copy()
    mask = downside_diff > 0
    downside_diff[mask] = 0.0
    squares = np.square(downside_diff)
    mean_squares = nanmean(squares, axis=0)
    dside_risk = np.sqrt(mean_squares) * np.sqrt(ann_factor)

    if len(returns.shape) == 2 and isinstance(returns, pd.DataFrame):
        dside_risk = pd.Series(dside_risk, index=returns.columns)
    return dside_risk


def information_ratio(returns, factor_returns):
    """
    Determines the Information ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
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

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = nanstd(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return nanmean(active_return) / tracking_error


def _aligned_series(*many_series):
    """
    Return a new list of series containing the data in the input series, but
    with their indices aligned. NaNs will be filled in for missing values.

    Parameters
    ----------
    many_series : list[pd.Series]

    Returns
    -------
    aligned_series : list[pd.Series]

        A new list of series containing the data in the input series, but
        with their indices aligned. NaNs will be filled in for missing values.

    """
    return [series
            for col, series in iteritems(pd.concat(many_series, axis=1))]


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
    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan, np.nan

    return alpha_beta_aligned(*_aligned_series(returns, factor_returns),
                              risk_free=risk_free, period=period,
                              annualization=annualization)


def alpha_beta_aligned(returns, factor_returns, risk_free=0.0, period=DAILY,
                       annualization=None):
    """Calculates annualized alpha and beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
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
    b = beta_aligned(returns, factor_returns, risk_free)
    a = alpha_aligned(returns, factor_returns, risk_free, period,
                      annualization, _beta=b)
    return a, b


def alpha(returns, factor_returns, risk_free=0.0, period=DAILY,
          annualization=None, _beta=None):
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
    _beta : float, optional
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.

    Returns
    -------
    float
        Alpha.
    """
    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan

    return alpha_aligned(*_aligned_series(returns, factor_returns),
                         risk_free=risk_free, period=period,
                         annualization=annualization, _beta=_beta)


def alpha_aligned(returns, factor_returns, risk_free=0.0, period=DAILY,
                  annualization=None, _beta=None):
    """Calculates annualized alpha.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
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
    _beta : float, optional
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.

    Returns
    -------
    float
        Alpha.
    """
    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    if _beta is None:
        _beta = beta_aligned(returns, factor_returns, risk_free)

    adj_returns = _adjust_returns(returns, risk_free)
    adj_factor_returns = _adjust_returns(factor_returns, risk_free)
    alpha_series = adj_returns - (_beta * adj_factor_returns)

    return nanmean(alpha_series) * ann_factor


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
    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan

    return beta_aligned(*_aligned_series(returns, factor_returns),
                        risk_free=risk_free)


def beta_aligned(returns, factor_returns, risk_free=0.0):
    """Calculates beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
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

    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan
    # Filter out dates with np.nan as a return value
    joint = np.vstack([_adjust_returns(returns, risk_free),
                       factor_returns])
    joint = joint[:, ~np.isnan(joint).any(axis=0)]
    if joint.shape[1] < 2:
        return np.nan

    cov = np.cov(joint, ddof=0)

    if np.absolute(cov[1, 1]) < 1.0e-30:
        return np.nan

    return cov[0, 1] / cov[1, 1]


def stability_of_timeseries(returns):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    float
        R-squared.

    """
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)),
                            cum_log_returns)[2]

    return rhat ** 2


def tail_ratio(returns):
    """Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    float
        tail ratio

    """

    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)
    # Be tolerant of nan's
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))


def cagr(returns, period=DAILY, annualization=None):
    """
    Compute compound annual growth rate.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
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
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns(np.asanyarray(returns), starting_value=1)[-1]

    return ending_value ** (1. / no_years) - 1


def beta_fragility_heuristic(returns, factor_returns):
    """
    Estimate fragility to drops in beta.
    A negative return value indicates potential losses
    could follow volatility in beta.
    The magnitude of the negative value indicates the size of
    the potential loss.

    seealso::

    `A New Heuristic Measure of Fragility and
Tail Risks: Application to Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
        An IMF Working Paper describing the heuristic

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float, np.nan
        The beta fragility of the strategy.

    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    return beta_fragility_heuristic_aligned(
        *_aligned_series(returns, factor_returns))


def beta_fragility_heuristic_aligned(returns, factor_returns):
    """
    Estimate fragility to drops in beta

    seealso::

    `A New Heuristic Measure of Fragility and
Tail Risks: Application to Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
        An IMF Working Paper describing the heuristic

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float, np.nan
        The beta fragility of the strategy.

    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    # combine returns and factor returns into pairs
    returns_series = pd.Series(returns)
    factor_returns_series = pd.Series(factor_returns)
    pairs = pd.concat([returns_series, factor_returns_series], axis=1)
    pairs.columns = ['returns', 'factor_returns']

    # exclude any rows where returns are nan
    pairs = pairs.dropna()
    # sort by beta
    pairs = pairs.sort_values(by='factor_returns')

    # find the three vectors, using median of 3
    start_index = 0
    mid_index = int(np.around(len(pairs) / 2, 0))
    end_index = len(pairs) - 1

    (start_returns, start_factor_returns) = pairs.iloc[start_index]
    (mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
    (end_returns, end_factor_returns) = pairs.iloc[end_index]

    factor_returns_range = (end_factor_returns - start_factor_returns)
    start_returns_weight = 0.5
    end_returns_weight = 0.5

    # find weights for the start and end returns
    # using a convex combination
    if not factor_returns_range == 0:
        start_returns_weight = \
            (mid_factor_returns - start_factor_returns) / \
            factor_returns_range
        end_returns_weight = \
            (end_factor_returns - mid_factor_returns) / \
            factor_returns_range

    # calculate fragility heuristic
    heuristic = (start_returns_weight*start_returns) + \
        (end_returns_weight*end_returns) - mid_returns

    return heuristic


def gpd_risk_estimates(returns, var_p=0.01):
    """
    Estimate VaR and ES using the Generalized Pareto Distribution (GPD)

    seealso::

    `An Application of Extreme Value Theory for
Measuring Risk <https://link.springer.com/article/10.1007/s10614-006-9025-7>`
        A paper describing how to use the Generalized Pareto
        Distribution to estimate VaR and ES.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    var_p : float
        The percentile to use for estimating the VaR and ES

    Returns
    -------
    [threshold, scale_param, shape_param, var_estimate, es_estimate]
        : list[float]
        threshold - the threshold use to cut off exception tail losses
        scale_param - a parameter (often denoted by sigma, capturing the
            scale, related to variance)
        shape_param - a parameter (often denoted by xi, capturing the shape or
            type of the distribution)
        var_estimate - an estimate for the VaR for the given percentile
        es_estimate - an estimate for the ES for the given percentile
    """
    if len(returns) < 3:
        result = np.array([0, 0, 0, 0])
        if isinstance(returns, pd.Series):
            result = pd.Series(result)
        return result
    return gpd_risk_estimates_aligned(*_aligned_series(returns, var_p))


def gpd_risk_estimates_aligned(returns, var_p=0.01):
    """
    Estimate VaR and ES using the Generalized Pareto Distribution (GPD)

    seealso::

    `An Application of Extreme Value Theory for
Measuring Risk <https://link.springer.com/article/10.1007/s10614-006-9025-7>`
        A paper describing how to use the Generalized Pareto
        Distribution to estimate VaR and ES.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    var_p : float
        The percentile to use for estimating the VaR and ES

    Returns
    -------
    [threshold, scale_param, shape_param, var_estimate, es_estimate]
        : list[float]
        threshold - the threshold use to cut off exception tail losses
        scale_param - a parameter (often denoted by sigma, capturing the
            scale, related to variance)
        shape_param - a parameter (often denoted by xi, capturing the shape or
            type of the distribution)
        var_estimate - an estimate for the VaR for the given percentile
        es_estimate - an estimate for the ES for the given percentile
    """
    result = np.array([0.0, 0.0, 0.0, 0.0])
    if not len(returns) < 3:

        DEFAULT_THRESHOLD = 0.2
        MINIMUM_THRESHOLD = 0.000000001
        returns_array = pd.Series(returns).as_matrix()
        flipped_returns = -1 * returns_array
        filtered_returns = flipped_returns[flipped_returns > 0]
        threshold = DEFAULT_THRESHOLD
        finished = False
        scale_param = 0
        shape_param = 0
        while not finished and threshold > MINIMUM_THRESHOLD:
            iteration_returns = \
                filtered_returns[filtered_returns >= threshold]
            param_result = \
                gpd_loglikelihood_minimizer_aligned(iteration_returns)
            if (param_result[0] is not False and
                    param_result[1] is not False):
                scale_param = param_result[0]
                shape_param = param_result[1]
                # non-negative shape parameter is required for fat tails
                if (shape_param > 0):
                    finished = True
            threshold = threshold / 2
        if (finished):
            var_estimate = gpd_var_calculator(threshold, scale_param,
                                              shape_param, var_p,
                                              len(returns_array),
                                              len(iteration_returns))
            es_estimate = gpd_es_calculator(var_estimate, threshold,
                                            scale_param, shape_param)
            result = np.array([threshold, scale_param, shape_param,
                               var_estimate, es_estimate])
    if isinstance(returns, pd.Series):
        result = pd.Series(result)
    return result


def gpd_es_calculator(var_estimate, threshold, scale_param,
                      shape_param):
    result = 0
    if ((1 - shape_param) != 0):
        result = (var_estimate/(1 - shape_param)) + \
            ((scale_param - (shape_param * threshold)) /
                (1 - shape_param))
    return result


def gpd_var_calculator(threshold, scale_param, shape_param,
                       probability, total_n, exceedance_n):
    result = 0
    if (exceedance_n > 0 and shape_param > 0):
        result = threshold+((scale_param / shape_param) *
                            (pow((total_n/exceedance_n) *
                             probability, -shape_param) - 1))
    return result


def gpd_loglikelihood_minimizer_aligned(price_data):
    result = [False, False]
    DEFAULT_SCALE_PARAM = 1
    DEFAULT_SHAPE_PARAM = 1
    if (len(price_data) > 0):
        gpd_loglikelihood_lambda = \
            gpd_loglikelihood_factory(price_data)
        optimization_results = \
            optimize.minimize(gpd_loglikelihood_lambda,
                              [DEFAULT_SCALE_PARAM,
                               DEFAULT_SHAPE_PARAM],
                              method='Nelder-Mead')
        if optimization_results.success:
            resulting_params = optimization_results.x
            if len(resulting_params) == 2:
                result[0] = resulting_params[0]
                result[1] = resulting_params[1]
    return result


def gpd_loglikelihood_factory(price_data):
    return lambda params: gpd_loglikelihood(params, price_data)


def gpd_loglikelihood(params, price_data):
    if (params[1] != 0):
        return -gpd_loglikelihood_scale_and_shape(params[0],
                                                  params[1],
                                                  price_data)
    else:
        return -gpd_loglikelihood_scale_only(params[0], price_data)


def gpd_loglikelihood_scale_and_shape_factory(price_data):
    # minimize a function of two variables requires a list of params
    # we are expecting the lambda below to be called as follows:
    # parameters = [scale, shape]
    # the final outer negative is added because scipy only minimizes
    return lambda params: \
        -gpd_loglikelihood_scale_and_shape(params[0],
                                           params[1],
                                           price_data)


def gpd_loglikelihood_scale_and_shape(scale, shape, price_data):
    n = len(price_data)
    result = -1 * float_info.max
    if (scale != 0):
        param_factor = shape / scale
        if (shape != 0 and param_factor >= 0 and scale >= 0):
            result = ((-n * np.log(scale)) -
                      (((1 / shape) + 1) *
                       (np.log((shape / scale * price_data) + 1)).sum()))
    return result


def gpd_loglikelihood_scale_only_factory(price_data):
    # the negative is added because scipy only minimizes
    return lambda scale: \
        -gpd_loglikelihood_scale_only(scale, price_data)


def gpd_loglikelihood_scale_only(scale, price_data):
    n = len(price_data)
    data_sum = price_data.sum()
    result = -1 * float_info.max
    if (scale >= 0):
        result = ((-n*np.log(scale)) - (data_sum/scale))
    return result


SIMPLE_STAT_FUNCS = [
    cum_returns_final,
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
    cagr,
    beta_fragility_heuristic,
    gpd_risk_estimates,
]

FACTOR_STAT_FUNCS = [
    information_ratio,
    alpha,
    beta,
]
