import pandas as pd
from data import get_prices
from returns import log_returns
from backtest import exception_series, kupiec_pof_test
from plots import plot_var_backtest
from scipy.stats import norm

def main():
    """
    rolling 1 day VaR backtest for single asset
    evals historical and parametric VaR using a fixed rolling window and valdiates models using POF
    """
    
    ticker = "SPY" # single asset ticker
    start = "2015-01-01" # start date for historical data
    alpha = 0.99 # confidence level
    window = 250 # ~ 1 trading year

    prices = get_prices([ticker], start=start) # download adjusted price data for the asset
    r = log_returns(prices)[ticker] # compute log returns and extract the single asset Series

    
    # shift(1) so VaR for day t only uses returns through t-1 (no look-ahead)
    var_hist = -r.rolling(window).quantile(1 - alpha).shift(1) # compute rolling empirical (1 - alpha) quantile of returns

    z = norm.ppf(1 - alpha) # z score corresponding to the left tail of the Normal distribution
    mu = r.rolling(window).mean().shift(1) # rolling mean of returns
    sigma = r.rolling(window).std(ddof=1).shift(1) # rolling sd
    var_param = -(mu + z * sigma) # VaR = -(mu + z * sd) parametric VaR form

    out = pd.DataFrame({ # combine realized losses and VaR estimates
        "Loss": -r, # realized losses (positive)
        "VaR_hist": var_hist,
        "VaR_param": var_param
    }).dropna()

    # ID VaR breaches for each model
    exc_hist = exception_series(r, out["VaR_hist"])
    exc_param = exception_series(r, out["VaR_param"])

    
    print(
        f"\nSingle-Asset VaR Backtest: {ticker} | "
        f"alpha={alpha} | window={window}"
    )

    # test whether exception freq matches expectation
    print("Historical:", kupiec_pof_test(exc_hist, alpha))
    print("Parametric:", kupiec_pof_test(exc_param, alpha))

    # plot realized losses against rolling VaR estimates
    plot_var_backtest(out, f"Rolling 1-Day VaR Backtest: {ticker} (alpha={alpha}, window={window})")

# only runs when executed directly
if __name__ == "__main__":

    main()