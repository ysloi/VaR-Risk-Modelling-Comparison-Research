"""
Tests for the README open question:

    Why does Parametric Normal (rolling std) pass at 95% while a normal
    quantile with GARCH volatility fails?

Run from the project folder, next to z_diagnostics.py, data.py, returns.py:

    python open_question_tests.py

Three experiments, all scored on the same common sample (the days where every
input exists), with the same walk-forward settings as the dashboard.

  1. Grid: volatility source x mean choice, normal quantile throughout.
     Separates "which sigma" from "which mean". The cell (rolling 250d sd,
     rolling 250d mean) is the dashboard's Parametric Normal and the cell
     (GARCH sigma, GARCH mean) is the dashboard's GARCH(1,1), so those two
     should reproduce the dashboard counts. The extra row (GARCH sigma +
     empirical quantile) reproduces the dashboard's FHS with GJR off.
  2. Out-of-sample calibration of each sigma: are returns / sigma too wide?
  3. Regime split: breaches by volatility tercile (calm / middle / turbulent), with the
     terciles defined three ways: GARCH's own sigma, trailing 20-day realised vol
     (known ex ante) and forward 20-day realised vol (ex-post label). Includes a
     Kupiec p-value within each regime, which a single overall Kupiec cannot give.
"""

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

from z_diagnostics import _garch_fit_params, _garch_sigma2

WINDOW = 1000      # GARCH fit window and long rolling window (days)
STEP = 20          # GARCH refit frequency (days)
ALPHAS = (0.95, 0.99)


# ── helpers ──────────────────────────────────────────────────────────────────

def kupiec(x: int, n: int, alpha: float) -> tuple[float, float]:
    """Kupiec POF likelihood ratio and p-value."""
    p, phat = 1 - alpha, x / n

    def ll(q):
        return xlogy(n - x, 1 - q) + xlogy(x, q)

    lr = float(-2 * (ll(p) - ll(phat)))
    return lr, float(1 - chi2.cdf(lr, 1))


def garch_walk_forward(port_r: pd.Series, window: int, step: int, alphas, o: int = 0):
    """
    Walk-forward GARCH, same logic as the dashboard.
    Returns a DataFrame with the one-step-ahead mean and sigma (return units)
    and the empirical standardised-residual quantile for each alpha (for FHS),
    plus the number of failed refit blocks.
    """
    x = port_r.to_numpy(dtype=float) * 100.0
    n = len(x)
    out = {"mu_garch": np.full(n, np.nan), "sig_garch": np.full(n, np.nan)}
    for a in alphas:
        out[f"qz_{a}"] = np.full(n, np.nan)
    fails = 0

    for i in range(window, n, step):
        end = min(i + step, n)
        try:
            par = _garch_fit_params(x[i - window:i], o)
        except Exception:
            fails += 1
            continue
        s2 = _garch_sigma2(x[i - window:end], par, window)
        z = (x[i - window:i] - par["mu"]) / np.sqrt(s2[:window])
        z = z[np.isfinite(z)]
        out["mu_garch"][i:end] = par["mu"] / 100.0
        out["sig_garch"][i:end] = np.sqrt(s2[window:]) / 100.0
        for a in alphas:
            out[f"qz_{a}"][i:end] = np.quantile(z, 1 - a)

    return pd.DataFrame(out, index=port_r.index), fails


def build_inputs(port_r: pd.Series, window: int, step: int, alphas) -> pd.DataFrame:
    """All inputs on one frame; rolling quantities are shifted so day t uses data to t-1."""
    g, fails = garch_walk_forward(port_r, window, step, alphas)
    d = pd.DataFrame({"r": port_r})
    d["sig_roll250"] = port_r.rolling(250).std(ddof=1).shift(1)
    d["sig_roll1000"] = port_r.rolling(window).std(ddof=1).shift(1)
    d["mu_roll250"] = port_r.rolling(250).mean().shift(1)
    d = d.join(g)
    d["zero"] = 0.0
    n_all = len(d)
    d = d.dropna()
    # neutral regime labels (not used in the experiment 1 sample, so no days are dropped)
    d["rv_back"] = port_r.rolling(20).std().shift(1).reindex(d.index)
    d["rv_fwd"] = port_r.rolling(20).std().shift(-20).reindex(d.index)
    print(f"Common sample: {len(d):,} days ({d.index[0].date()} to {d.index[-1].date()}), "
          f"{n_all - len(d):,} days dropped, failed GARCH refit blocks: {fails}")
    return d


SIGMAS = {
    "rolling sd 250d":  "sig_roll250",
    "rolling sd 1000d": "sig_roll1000",
    "GARCH sigma_t":    "sig_garch",
}
MEANS = {
    "rolling mean 250d": "mu_roll250",
    "GARCH mean":        "mu_garch",
    "zero":              "zero",
}


def breaches(d: pd.DataFrame, var: pd.Series) -> np.ndarray:
    return ((-d["r"]) > var).to_numpy()


# ── experiment 1: grid ───────────────────────────────────────────────────────

def experiment_grid(d: pd.DataFrame) -> None:
    n = len(d)
    print("\n" + "=" * 78)
    print("EXPERIMENT 1: volatility source x mean, normal quantile")
    print("=" * 78)
    for alpha in ALPHAS:
        z = norm.ppf(1 - alpha)
        print(f"\nalpha = {alpha:.2f}   expected exceptions = {n * (1 - alpha):.0f}   "
              f"(cell = exceptions, Kupiec p; * = p < 0.05)")
        table = {}
        for sname, scol in SIGMAS.items():
            row = {}
            for mname, mcol in MEANS.items():
                var = -(d[mcol] + z * d[scol])
                x = int(breaches(d, var).sum())
                _, p = kupiec(x, n, alpha)
                row[mname] = f"{x} (p={p:.3f}){'*' if p < 0.05 else ''}"
            table[sname] = row
        print(pd.DataFrame(table).T.to_string())

        # empirical quantile on GARCH sigma = FHS (GJR off)
        var = -(d["mu_garch"] + d["sig_garch"] * d[f"qz_{alpha}"])
        x = int(breaches(d, var).sum())
        _, p = kupiec(x, n, alpha)
        print(f"GARCH sigma + GARCH mean + EMPIRICAL quantile (= FHS, GJR off): "
              f"{x} (p={p:.3f}){'*' if p < 0.05 else ''}")

    print("\nReproduction check against the dashboard (same window/refit/sample only):")
    print("  Parametric Normal = cell (rolling sd 250d, rolling mean 250d)")
    print("  GARCH(1,1)        = cell (GARCH sigma_t, GARCH mean)")


# ── experiment 2: out-of-sample calibration of sigma ─────────────────────────

def experiment_calibration(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("EXPERIMENT 2: out-of-sample calibration of each sigma (zero mean)")
    print("=" * 78)
    print("Statistics of r_t / sigma_t on the common sample. If sigma is well "
          "calibrated and the\nreturns were normal: mean square = 1, q05 = -1.645, "
          "q01 = -2.326.\n")
    rows = {}
    for sname, scol in SIGMAS.items():
        u = d["r"] / d[scol]
        rows[sname] = {
            "mean sq": float((u ** 2).mean()),
            "q05": float(np.quantile(u, 0.05)),
            "q01": float(np.quantile(u, 0.01)),
            "skew": float(u.skew()),
            "avg sigma %": float(d[scol].mean() * 100),
        }
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("\nmean sq > 1 : sigma too small on average (under-forecast).")
    print("q05 further out than -1.645 : the left tail is fatter than the normal "
          "quantile allows, whatever the average level of sigma.")


# ── experiment 3: regime split ───────────────────────────────────────────────

REGIME_SCHEMES = {
    "GARCH sigma_t (its own forecast)":                          "sig_garch",
    "trailing 20d realised vol (known at t-1)":                  "rv_back",
    "forward 20d realised vol (days t+1..t+20, ex-post label)":  "rv_fwd",
}


def experiment_regimes(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("EXPERIMENT 3: breaches by volatility regime, three ways of defining the regime")
    print("=" * 78)
    print("Parametric = 250d sd + 250d mean; GARCH = GARCH sigma + GARCH mean (normal quantile).")
    print("p_* = Kupiec p-value computed WITHIN the regime (target hit rate = 1 - alpha).")
    for label, col in REGIME_SCHEMES.items():
        sub = d.dropna(subset=[col]).copy()
        sub["regime"] = pd.qcut(sub[col], 3, labels=["calm", "middle", "turbulent"])
        print("\n" + "-" * 78)
        print(f"Regime defined by: {label}   ({len(sub):,} days)")
        for alpha in ALPHAS:
            z = norm.ppf(1 - alpha)
            var_p = -(sub["mu_roll250"] + z * sub["sig_roll250"])
            var_g = -(sub["mu_garch"] + z * sub["sig_garch"])
            sub["b_param"] = breaches(sub, var_p)
            sub["b_garch"] = breaches(sub, var_g)
            sub["v_param"] = var_p * 100
            sub["v_garch"] = var_g * 100
            g = sub.groupby("regime", observed=True).agg(
                days=("r", "size"),
                hit_param=("b_param", "mean"),
                hit_garch=("b_garch", "mean"),
                avg_var_param_pct=("v_param", "mean"),
                avg_var_garch_pct=("v_garch", "mean"),
                x_param=("b_param", "sum"),
                x_garch=("b_garch", "sum"),
            )
            g["p_param"] = [kupiec(int(x), int(n), alpha)[1] for x, n in zip(g["x_param"], g["days"])]
            g["p_garch"] = [kupiec(int(x), int(n), alpha)[1] for x, n in zip(g["x_garch"], g["days"])]
            g = g.drop(columns=["x_param", "x_garch"])
            print(f"\nalpha = {alpha:.2f}   target hit rate = {1 - alpha:.3f}")
            print(g.round(4).to_string())
    print("\nSampling noise: with ~650 days per regime, a hit rate at 95% has a "
          "standard deviation of\nabout 0.85 percentage points (about 0.4 at 99%), "
          "so read gaps below that as noise.")
    print("A pattern that appears under all three regime definitions is more credible than "
          "one that\nappears only under the GARCH-sigma definition.")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data import get_prices
    from returns import log_returns, portfolio_returns, normalize_weights

    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    weights = normalize_weights(np.array([0.4, 0.3, 0.2, 0.1]))
    prices = get_prices(tickers, start="2015-01-01")
    port_r = portfolio_returns(log_returns(prices), weights)

    d = build_inputs(port_r, WINDOW, STEP, ALPHAS)
    experiment_grid(d)
    experiment_calibration(d)
    experiment_regimes(d)