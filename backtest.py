import numpy as np
import pandas as pd
from scipy.stats import chi2

def exception_series(realized_returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    computes a VaR exception (also called a 'hit') series.

    an exception occurs when the realized loss exceeds the VaR estimate:
    loss > VaR <-> -r_t > VaR_t

    returns:
        a pandas series of 1s and 0s
        1 -> exception occured
        0 -> no exception
    """
    # combine the realized returns and VaR series into a single DataFrame
    # this aligns them by date/index
    aligned = pd.concat([realized_returns, var_series], axis=1).dropna()
    r = aligned.iloc[:, 0] # extract the realized returns (first column)
    v = aligned.iloc[:, 1] # extract the VaR estimates (second column)
    return (-r > v).astype(int) # checks whether the realized loss exceeded VaR
    # -r converts returns into losses
    # (-r > v) produces a bool Series
    # .astype(int) converts T or F into 1 or 0 respectively

def kupiec_pof_test(exceptions: pd.Series, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures (POF )test

    this tests whether the observed exception frequency matches the expected
    exception probability implied by the VaR confidence level.

    hypotheses:
    H0: P(exception) = 1 - alpha
    H1: P(exception) != 1 - alpha

    this returns a dicttionary containing test statistic, p-value, and summary metrics
    """
    exc = exceptions.dropna().to_numpy() # removes missing values and converts the expection Series into a NumPy array
    x = int(exc.sum()) # count total number of expections (sum of 1's)
    n = int(len(exc)) # total of observations 
    p = 1 - alpha # expected exception probability under null hypothesis
    # example: alpha = 0.99 -> p = 0.01

    if n == 0: # no data means test cannon be computed
        raise ValueError("No exceptions to test (empty series).")

    # if zero exceptions or exceptions every period
    # likelihood ratio formula breaks
    if x == 0 or x == n:
        return {
            "LR_pof": float("inf"), 
            "p_value": 0.0, 
            "exceptions": x, 
            "n": n}

    phat = x / n # exception rate observed in data

    # likelihood ratio for POF test
    # likelihood under null hypothesis and under empirical probability (p and phat)
    lr = -2 * (
        (n - x) * np.log((1 - p) / (1 - phat)) +
        x * np.log(p / phat)
    )

    # compute the p-value, degree of freedom = 1
    pval = 1 - chi2.cdf(lr, df=1)


    return {
            "LR_pof": float(lr), # likelihood ratio stat
            "p_value": float(pval), # stat significance
            "exceptions": x, # num of VaR breaches
            "n": n, # num of oberservations
            "hit_rate": phat} # exception freq


def christoffersen_independence_test(exceptions: pd.Series) -> dict:
    """
    Christoffersen (1998) independence test.

    Tests whether VaR exceptions cluster in time (are serially dependent).
    Compares an i.i.d. Bernoulli model (H0) to a first-order Markov chain (H1).

    Transition counts from consecutive pairs (I_{t-1}, I_t):
        T00  no exception followed by no exception
        T01  no exception followed by exception
        T10  exception followed by no exception
        T11  exception followed by exception (clustering)

    LR_ind ~ chi2(df=1) under H0.

    returns dict with LR_ind, p_value, transition counts, and conditional
    exception probabilities pi_01 and pi_11.
    """
    exc = exceptions.dropna().to_numpy()
    n = len(exc)

    if n < 2:
        raise ValueError("Need at least 2 observations for independence test.")

    # build 2x2 transition counts from consecutive pairs
    T00 = int(((exc[:-1] == 0) & (exc[1:] == 0)).sum())
    T01 = int(((exc[:-1] == 0) & (exc[1:] == 1)).sum())
    T10 = int(((exc[:-1] == 1) & (exc[1:] == 0)).sum())
    T11 = int(((exc[:-1] == 1) & (exc[1:] == 1)).sum())

    base = {"T00": T00, "T01": T01, "T10": T10, "T11": T11}

    # degenerate cases: cannot estimate one or both rows of the transition matrix
    row0_total = T00 + T01
    row1_total = T10 + T11
    if row0_total == 0 or row1_total == 0 or (T01 + T11) == 0:
        return {"LR_ind": float("inf"), "p_value": 0.0, "pi_01": float("nan"),
                "pi_11": float("nan"), **base}

    pi_01 = T01 / row0_total   # P(exception | prev: no exception)
    pi_11 = T11 / row1_total   # P(exception | prev: exception)

    # degenerate: one conditional probability is 0 or 1 → log blows up
    if pi_01 in (0.0, 1.0) or pi_11 in (0.0, 1.0):
        return {"LR_ind": float("inf"), "p_value": 0.0,
                "pi_01": float(pi_01), "pi_11": float(pi_11), **base}

    # unconditional exception probability under H0 (i.i.d.)
    p_hat = (T01 + T11) / (T00 + T01 + T10 + T11)

    if p_hat in (0.0, 1.0):
        return {"LR_ind": float("inf"), "p_value": 0.0,
                "pi_01": float(pi_01), "pi_11": float(pi_11), **base}

    # LR = -2 * [log L(H0) - log L(H1)]
    ll_h0 = (T00 + T10) * np.log(1 - p_hat) + (T01 + T11) * np.log(p_hat)
    ll_h1 = (T00 * np.log(1 - pi_01) + T01 * np.log(pi_01)
             + T10 * np.log(1 - pi_11) + T11 * np.log(pi_11))
    lr = -2 * (ll_h0 - ll_h1)

    pval = 1 - chi2.cdf(lr, df=1)

    return {
        "LR_ind":  float(lr),
        "p_value": float(pval),
        "pi_01":   float(pi_01),
        "pi_11":   float(pi_11),
        **base,
    }


def christoffersen_cc_test(exceptions: pd.Series, alpha: float) -> dict:
    """
    Christoffersen (1998) conditional coverage test.

    Combines Kupiec POF (unconditional coverage) and the independence test
    into a joint test:
        LR_cc = LR_pof + LR_ind  ~  chi2(df=2)

    Rejects when either the exception frequency or the serial clustering of
    exceptions is inconsistent with the model's stated confidence level.

    returns dict with LR_cc, p_value, and all sub-test statistics.
    """
    pof = kupiec_pof_test(exceptions, alpha)
    ind = christoffersen_independence_test(exceptions)

    lr_pof = pof["LR_pof"]
    lr_ind = ind["LR_ind"]

    if lr_pof == float("inf") or lr_ind == float("inf"):
        return {
            "LR_cc":      float("inf"),
            "p_value":    0.0,
            "LR_pof":     lr_pof,
            "LR_ind":     lr_ind,
            "exceptions": pof["exceptions"],
            "n":          pof["n"],
            "hit_rate":   pof.get("hit_rate", float("nan")),
            "pi_01":      ind["pi_01"],
            "pi_11":      ind["pi_11"],
        }

    lr_cc = lr_pof + lr_ind
    pval  = 1 - chi2.cdf(lr_cc, df=2)

    return {
        "LR_cc":      float(lr_cc),
        "p_value":    float(pval),
        "LR_pof":     float(lr_pof),
        "LR_ind":     float(lr_ind),
        "exceptions": pof["exceptions"],
        "n":          pof["n"],
        "hit_rate":   pof["hit_rate"],
        "pi_01":      ind["pi_01"],
        "pi_11":      ind["pi_11"],
    }