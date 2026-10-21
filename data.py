import pandas as pd
import yfinance as yf
from typing import Optional

def get_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    downloads adjusted historical price data for one or more tickers (single asset vs port)

    parameters
        tickers : list[str]
            list of ticker symbols (e.g., ["SPY", "QQQ", "TLT"])
        start : str
            Start date for historical data (YYYY-MM-DD)
        end : str | None
            End date (exclusive, YYYY-MM-DD); None fetches through the latest close
    returns
            pandas DataFrame indexed by date with one column per ticker
            containing adjusted closing prices.
    """
    df = yf.download( #downloading price data from Yahoo Finance (yfinance)
        tickers, # individual stocks
        start=start, # start date
        end=end, # end date (None = latest available)
        auto_adjust=True,   # ensures prices are total return consistent
        progress=False
    )


    # edge case, if yf returns no data, raise exception
    if df.empty:
        raise ValueError("No data returned. Check tickers/start date.")

    # yfinance returns multi-index columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"]
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"No data returned for: {missing}")
        # yfinance sorts columns alphabetically; restore the requested order
        # so portfolio weights line up with the tickers they were written for
        prices = prices[tickers].copy()
    # only activates else statement if only a single ticker is requested (single-level column)
    else:
        prices = df[["Close"]].copy()
        # rename column to match ticker name, makes sure output format is consistent
        prices.columns = tickers

    prices = prices.dropna(how="all") # drops rows where all tickers are missing prices (non-trading days or partial data)

    return prices
