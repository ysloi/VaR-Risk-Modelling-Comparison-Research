import numpy as np
import pandas as pd

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    computes log returns from price levels

    log returns form
        r_t = ln(P_t / P_{t-1})

    parameters
        prices: pd.DataFrame
            price levels indexed by date

    returns
        pd.DateFrame
            log returns with the same columns
    
    """
    return np.log(prices / prices.shift(1)).dropna() 
    # moves prices down by one row
    # aligns each prices with previous period value
    # computes gross returns    
    # converts gross returns into log returns
    # removes first row

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    normalizes port weights so they = 1

    parameters
        weights: np.ndarray
            raw port weights

    returns
        np.ndarray
            normalized weights summing to 1
    """
    
    w = np.asarray(weights, dtype=float) # convert input to NumPy array of floats
    if w.ndim != 1: # ensure weights are 1D, should be a vector, not a matrix
        raise ValueError("weights must be 1D") # throw an execption
    s = w.sum() # sum of weights
    if s == 0: # prevent / by zero, zero sum port bad
        raise ValueError("weights sum to 0")
    return w / s # norm weights so total exposure = 1 

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    computes port returns from asset returns and weights

    parameters
        returns: pd.DataFrame
            asset return matrix T x N
        weights: np.ndarray
            port weights length N

    returns
        pd.Series
            port return time series length T
    """
    
    w = normalize_weights(weights) # norm weights make sure = 1
    if returns.shape[1] != len(w): # ensure num of assets matches num of weights
        raise ValueError("weights length must match number of assets") # throw exception

    return returns @ w # matrix multiplication, each row computes weighted sum of asset returns