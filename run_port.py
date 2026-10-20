import pandas as pd
from data import get_prices
from returns import log_returns, portfolio_returns
from models import var_monte_carlo_portfolio
from backtest import exception_series, kupiec_pof_test
from plots import plot_var_backtest
from scipy.stats import norm

def main():
    """
    end to end rolling VaR backtesting pipeline for multi asset port

    steps
        download price data
        compute returns
        construct port returns
        compute rolling VaR (Hist, Para, MC)
        backtest VaR models using POF
        plot results
    """
    tickers = ["SPY", "QQQ", "TLT", "GLD"] # asset universe
    weights = [0.4, 0.3, 0.2, 0.1]   # port weights must align with tickers order
    start = "2015-01-01" # start date for historical data
    alpha = 0.99 # confidence level (99%)
    window = 250 # ~1 trading year

    prices = get_prices(tickers, start=start) # download adjusted price data
    rets = log_returns(prices) # convert price levels to log returns

    port_r = portfolio_returns(rets, weights) # compute port return series using normlized weights

    # shift(1) so VaR for day t only uses returns through t-1 (no look-ahead)
    var_hist = -port_r.rolling(window).quantile(1 - alpha).shift(1) # rolling empirical (1 - alpha) quantile of port returns
    z = norm.ppf(1 - alpha) # z score for parametric norm VaR
    mu = port_r.rolling(window).mean().shift(1) # rolling mean of port returns
    sigma = port_r.rolling(window).std(ddof=1).shift(1) # rolling vol (sample sd)
    var_param = -(mu + z * sigma) # parametric norm VaR form, VaR = -(mu + z * sd)

    var_mc = [] # declared list to store rolling Monte Carlo VaR values
    idx = [] # and dates
    
    for i in range(window, len(rets)): # loop through time using a rolling window
        w_rets = rets.iloc[i-window:i]  # extract rolling window of asset returns
        v = var_monte_carlo_portfolio(
            w_rets,
            weights,
            alpha=alpha, 
            n_sims=25_000,
            seed=42
        ) # compute Monte Carlo VaR for port
        
        var_mc.append(v) # store results
        idx.append(rets.index[i]) 

    var_mc = pd.Series(var_mc, index=idx, name="VaR_mc") # convert Monte Carlo to panda Series

    out = pd.DataFrame({ # combine realized losses and VaR estimates
        "Loss": -port_r, # realized losses (positive)
        "VaR_hist": var_hist,
        "VaR_param": var_param,
    }).join(var_mc, how="inner").dropna()

    # compute exception series for each VaR model
    exc_hist = exception_series(port_r, out["VaR_hist"])
    exc_param = exception_series(port_r, out["VaR_param"])
    exc_mc = exception_series(port_r, out["VaR_mc"])

    print(f"\nPortfolio VaR Backtest: {tickers} | weights={weights} | alpha={alpha} | window={window}")
    # freq backtesting for each VaR method
    print("Historical:", kupiec_pof_test(exc_hist, alpha))
    print("Parametric:", kupiec_pof_test(exc_param, alpha))
    print("MonteCarlo:", kupiec_pof_test(exc_mc, alpha))

    # visualization
    plot_var_backtest(
        out, 
        f"Rolling 1-Day Portfolio VaR (alpha={alpha}, window={window})"
    )

if __name__ == "__main__": # ensures main() only runs when script is run directly
    main()