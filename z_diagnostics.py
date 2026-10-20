"""
Diagnostics for why GARCH(1,1) with a normal quantile under-covers at 95%
while the rolling Parametric Normal does not.

Run from the project folder (the one containing dashboard.py, data.py, returns.py):

    python z_diagnostics.py

Do NOT import dashboard.py from here: it runs Streamlit code at import time.
The two GARCH helpers below are copies of the ones in dashboard.py.
"""

import warnings

import numpy as np
import pandas as pd


# ── copies of the dashboard helpers ──────────────────────────────────────────

def _garch_fit_params(x_pct: np.ndarray, o: int) -> dict:
    """Fit GARCH(1,o,1) with normal QMLE on percent returns; return parameters."""
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(x_pct, mean="Constant", vol="Garch", p=1, o=o, q=1,
                         dist="normal", rescale=False).fit(disp="off")
    if res.convergence_flag != 0:
        raise RuntimeError("no_convergence")
    p = res.params
    par = dict(
        mu=float(p["mu"]), omega=float(p["omega"]),
        alpha=float(p["alpha[1]"]), beta=float(p["beta[1]"]),
        gamma=float(p["gamma[1]"]) if o else 0.0,
    )
    if par["alpha"] + 0.5 * par["gamma"] + par["beta"] >= 1.0:
        raise RuntimeError("non_stationary")
    return par


def _garch_sigma2(x_pct: np.ndarray, par: dict, n_fit: int) -> np.ndarray:
    """Conditional variance recursion with fixed parameters (one-step-ahead)."""
    eps = x_pct - par["mu"]
    s2 = np.empty(len(x_pct))
    s2[0] = np.var(eps[:n_fit])
    om, a, g, b = par["omega"], par["alpha"], par["gamma"], par["beta"]
    for t in range(1, len(x_pct)):
        e = eps[t - 1]
        s2[t] = om + (a + (g if e < 0 else 0.0)) * e * e + b * s2[t - 1]
    return s2


# ── diagnostic ───────────────────────────────────────────────────────────────

def z_diagnostics(port_r: pd.Series, window: int = 1000, step: int = 20,
                  o: int = 0) -> pd.DataFrame:
    """
    For each refit window, fit GARCH and report properties of the in-sample
    standardised residuals z, plus the fitted mean vs the 250-day rolling mean.
    """
    x = port_r.to_numpy(dtype=float) * 100.0
    rows = []
    for i in range(window, len(x), step):
        try:
            par = _garch_fit_params(x[i - window:i], o)
        except Exception:
            continue
        s2 = _garch_sigma2(x[i - window:i], par, window)
        z = (x[i - window:i] - par["mu"]) / np.sqrt(s2)
        rows.append(dict(
            date=port_r.index[i],
            mu_garch=par["mu"] / 100,
            mu_roll=port_r.iloc[i - 250:i].mean(),
            var_z=z.var(),
            q05=np.quantile(z, 0.05),
            q01=np.quantile(z, 0.01),
            skew=pd.Series(z).skew(),
            persistence=par["alpha"] + par["beta"] + 0.5 * par["gamma"],
        ))
    return pd.DataFrame(rows).set_index("date")


def summarise(d: pd.DataFrame) -> None:
    pd.set_option("display.width", 120)
    print(f"\nRefit windows analysed: {len(d)}\n")
    print(d.describe().loc[["mean", "std", "min", "max"]].round(5).to_string())
    print("\nReference values:  normal q05 = -1.645,  normal q01 = -2.326,  "
          "var_z = 1,  skew = 0")
    print(f"\nMean var_z : {d['var_z'].mean():.3f}   (>1 means residuals are wider "
          "than the normal quantile assumes)")
    print(f"Mean q05   : {d['q05'].mean():.3f}   vs -1.645")
    print(f"Mean q01   : {d['q01'].mean():.3f}   vs -2.326")
    print(f"Mean skew  : {d['skew'].mean():.3f}")
    print(f"Mean (mu_garch - mu_roll): {(d['mu_garch'] - d['mu_roll']).mean():.5f} per day")


if __name__ == "__main__":
    # imported here so the functions above can be tested without the project files
    from data import get_prices
    from returns import log_returns, portfolio_returns, normalize_weights

    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    weights = normalize_weights(np.array([0.4, 0.3, 0.2, 0.1]))

    prices = get_prices(tickers, start="2015-01-01")
    port_r = portfolio_returns(log_returns(prices), weights)

    print("Running GARCH(1,1), o=0 ...")
    summarise(z_diagnostics(port_r, window=1000, step=20, o=0))