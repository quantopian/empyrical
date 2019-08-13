from collections import OrderedDict
import pandas as pd


def perf_attrib(returns,
                positions,
                factor_returns,
                factor_loadings):
    """
    Attributes the performance of a returns stream to a set of risk factors.

    Performance attribution determines how much each risk factor, e.g.,
    momentum, the technology sector, etc., contributed to total returns, as
    well as the daily exposure to each of the risk factors. The returns that
    can be attributed to one of the given risk factors are the
    `common_returns`, and the returns that _cannot_ be attributed to a risk
    factor are the `specific_returns`. The `common_returns` and
    `specific_returns` summed together will always equal the total returns.

    Parameters
    ----------
    returns : pd.Series
        Returns for each day in the date range.
        - Example:
            2017-01-01   -0.017098
            2017-01-02    0.002683
            2017-01-03   -0.008669

    positions: pd.Series
        Daily holdings in percentages, indexed by date.
        - Examples:
            dt          ticker
            2017-01-01  AAPL      0.417582
                        TLT       0.010989
                        XOM       0.571429
            2017-01-02  AAPL      0.202381
                        TLT       0.535714
                        XOM       0.261905

    factor_returns : pd.DataFrame
        Returns by factor, with date as index and factors as columns
        - Example:
                        momentum  reversal
            2017-01-01  0.002779 -0.005453
            2017-01-02  0.001096  0.010290

    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date and ticker as
        index, and factors as columns.
        - Example:
                               momentum  reversal
            dt         ticker
            2017-01-01 AAPL   -1.592914  0.852830
                       TLT     0.184864  0.895534
                       XOM     0.993160  1.149353
            2017-01-02 AAPL   -0.140009 -0.524952
                       TLT    -1.066978  0.185435
                       XOM    -1.798401  0.761549

    Returns
    -------
    tuple of (risk_exposures_portfolio, perf_attribution)

    risk_exposures_portfolio : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515

    perf_attribution : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
        - Example:
                        momentum  reversal  common_returns  specific_returns
            dt
            2017-01-01  0.249087  0.935925        1.185012          1.185012
            2017-01-02 -0.003194 -0.400786       -0.403980         -0.403980

    Note
    ----
    See https://en.wikipedia.org/wiki/Performance_attribution for more details.
    """

    # Make risk data match time range of returns
    start = returns.index[0]
    end = returns.index[-1]
    factor_returns = factor_returns.loc[start:end]
    factor_loadings = factor_loadings.loc[start:end]

    factor_loadings.index = factor_loadings.index.set_names(['dt', 'ticker'])

    positions = positions.copy()
    positions.index = positions.index.set_names(['dt', 'ticker'])

    risk_exposures_portfolio = compute_exposures(positions,
                                                 factor_loadings)

    perf_attrib_by_factor = risk_exposures_portfolio.multiply(factor_returns)
    common_returns = perf_attrib_by_factor.sum(axis='columns')

    tilt_exposure = risk_exposures_portfolio.mean()
    tilt_returns = factor_returns.multiply(tilt_exposure).sum(axis='columns')
    timing_returns = common_returns - tilt_returns
    specific_returns = returns - common_returns

    returns_df = pd.DataFrame(OrderedDict([
        ('total_returns', returns),
        ('common_returns', common_returns),
        ('specific_returns', specific_returns),
        ('tilt_returns', tilt_returns),
        ('timing_returns', timing_returns)
        ]))

    return (risk_exposures_portfolio,
            pd.concat([perf_attrib_by_factor, returns_df], axis='columns'))


def compute_exposures(positions, factor_loadings):
    """
    Compute daily risk factor exposures.

    Parameters
    ----------
    positions: pd.Series
        A series of holdings as percentages indexed by date and ticker.
        - Examples:
            dt          ticker
            2017-01-01  AAPL      0.417582
                        TLT       0.010989
                        XOM       0.571429
            2017-01-02  AAPL      0.202381
                        TLT       0.535714
                        XOM       0.261905

    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date and ticker as
        index, and factors as columns.
        - Example:
                               momentum  reversal
            dt         ticker
            2017-01-01 AAPL   -1.592914  0.852830
                       TLT     0.184864  0.895534
                       XOM     0.993160  1.149353
            2017-01-02 AAPL   -0.140009 -0.524952
                       TLT    -1.066978  0.185435
                       XOM    -1.798401  0.761549

    Returns
    -------
    risk_exposures_portfolio : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515
    """
    risk_exposures = factor_loadings.multiply(positions, axis='rows')
    return risk_exposures.groupby(level='dt').sum()
