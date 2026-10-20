import numpy as np
import pandas as pd
from scipy.stats import norm, genpareto

def var_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """
    historical 1 day VaR

    parameters
        r: pd.Series
            time series of returns
        alpha: float
            confidence level

    returns
        float
            positive VaR value represents loss threshold
    
    """
    q = np.quantile(r, 1 - alpha) #compute the (1- alpha) empirical quantile of returns
    # example: alpha = 0.99 (99% confidence) -> 1% left tail
    return float(-q) # quantile is typically negative
    # multiply by -1 so VaR is reported as a positive loss number

def cvar_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """
    historical conditional VaR aka expected shortfall

    this measures the average loss given the VaR threshold has already been breached

    parameters
        r: pd.Series
            historical returns
        alpha: float
            confidence level

    returns
        float
            positive expected loss beyond VaR
    
    """
    cutoff = np.quantile(r, 1 - alpha) # compute VaR cutoff (left-tail quantile)
    tail = r[r <= cutoff] # select returns that are worse than or equal VaR cutoff
    return float(-tail.mean()) # computes mean of tail losses, negate so CVaR is expressed as positive loss

def var_parametric_normal(r: pd.Series, alpha: float = 0.99) -> float:
    """
    parametric VaR assuming returns follow a norm distribution

    uses
        VaR = -(mu + z_alpha * sd)

    parameters
        r: pd.Series
            historical returns
        alpha: float
            confidence level
    """
    mu = r.mean() # mean return 
    sigma = r.std(ddof=1) # estimate sd, ddof=1 uses unbiased sample sd
    z = norm.ppf(1 - alpha)  # compute z-score linked to left tail, alpha=0.99 -> norm.ppf(0.01) ~ -2.33
    return float(-(mu + z * sigma)) # combine mean and vol into VaR formula, multiply by -1

def var_monte_carlo_portfolio(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo VaR for multi-asset port

    assumes returns follow multivariate norm distro with empirical mean and covariance from data

    parameters
        returns: pd.DataFrame
            T x N matrix of asset returns
        weights: np.ndarray
            port weights length of N
        alpha: float
            confidence level
        n_sims: int
            n of Monte Carlo simulations 
        seed: int
            random seed for reproduciblity

        returns:
            float
                port VaR positive loss format
    """
    rng = np.random.default_rng(seed) # NumPy rand num generator
    mu = returns.mean().to_numpy() # compute mean return vector length N
    cov = returns.cov().to_numpy() # compute covariance matrix N x N

    sims = rng.multivariate_normal(mu, cov, size=n_sims) # sim multivariate normal returns, output shape = n_sims, N
    w = np.asarray(weights, dtype=float) # converts weights to NumPy array
    w = w / w.sum() # normalize weights to ensure they sum to 1, protects against user input error 
    port = sims @ w  # compute simulated port returns, matrix multiplication
    q = np.quantile(port, 1 - alpha) # left-tail quantile of port returns

    return float(-q) # convert to positive loss value


def var_garch(r: pd.Series, alpha: float = 0.99) -> float:
    """
    1-day-ahead parametric VaR using GARCH(1,1) conditional volatility.

    Fits a GARCH(1,1) model with normal innovations on the return window r
    and uses the one-step-ahead variance forecast to estimate VaR:

        VaR = -(mu + z_alpha * sigma_{T+1|T})

    This captures volatility clustering that rolling-std ignores: after a
    calm period sigma is low; after a turbulent period sigma is high.

    parameters
        r: pd.Series
            historical return window (typically 250 trading days)
        alpha: float
            confidence level (e.g. 0.99 for 99% VaR)

    returns
        float
            positive VaR value (loss convention)
    """
    from arch import arch_model

    # arch works in percentage space for numerical stability
    scaled = r * 100
    am = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")

    # one-step-ahead variance forecast at horizon 1
    forecast = res.forecast(horizon=1, reindex=False)
    sigma_next = float(np.sqrt(forecast.variance.values[-1, 0])) / 100

    mu = float(r.mean())
    z  = norm.ppf(1 - alpha)
    return float(-(mu + z * sigma_next))


def var_evt_pot(
    r: pd.Series,
    alpha: float = 0.99,
    threshold_quantile: float = 0.95,
) -> float:
    """
    1-day VaR via Peaks-over-Threshold (POT) with Generalized Pareto Distribution.

    The POT method fits a GPD only to the extreme tail of the loss distribution,
    which is more efficient than fitting a parametric model to the whole sample.

    Steps
    -----
    1. Convert returns to losses: L = -r
    2. Choose threshold u = quantile(L, threshold_quantile)
    3. Extract exceedances: e = L[L > u] - u
    4. Fit GPD(xi, beta) to e using MLE with loc fixed at 0
    5. Invert the POT survival function to get VaR at level alpha:

           VaR = u + (beta/xi) * [(n/N_u * (1-alpha))^(-xi) - 1]   (xi ≠ 0)
           VaR = u + beta * log(n / (N_u * (1-alpha)))              (xi ≈ 0)

    parameters
        r: pd.Series
            historical return window
        alpha: float
            VaR confidence level (must satisfy threshold_quantile < alpha)
        threshold_quantile: float
            quantile of losses used as the GPD threshold (default 0.95)

    returns
        float
            positive VaR value (loss convention)

    raises
        ValueError
            if fewer than 10 exceedances remain after applying the threshold
    """
    losses = -r.to_numpy(dtype=float)
    u      = float(np.quantile(losses, threshold_quantile))

    exceedances = losses[losses > u] - u
    if len(exceedances) < 10:
        raise ValueError(
            f"Only {len(exceedances)} exceedances above threshold — need ≥ 10 for GPD fit."
        )

    # MLE fit with location fixed at 0; returns (shape=xi, loc=0, scale=beta)
    xi, _, beta = genpareto.fit(exceedances, floc=0)

    n   = len(losses)
    n_u = len(exceedances)

    # probability mass above u assigned to the POT tail
    p_exceed = (1 - alpha) * n / n_u   # fraction of exceedances needed

    if p_exceed <= 0 or p_exceed >= 1:
        raise ValueError(
            f"Invalid p_exceed={p_exceed:.4f}. Check threshold_quantile < alpha."
        )

    if abs(xi) < 1e-8:                             # exponential limit (xi → 0)
        var_evt = u + beta * np.log(1.0 / p_exceed)
    else:
        var_evt = u + (beta / xi) * (p_exceed ** (-xi) - 1.0)

    return float(var_evt)

def _fit_garch_fhs(
    r: pd.Series,
    p: int = 1,
    o: int = 0,
    q: int = 1,
):
    """
    Fit GARCH(p,o,q) with normal QMLE and return the pieces FHS needs.

    o=1 gives GJR-GARCH (leverage effect). dist='normal' is used only as a
    quasi-likelihood: the residual distribution is taken from the data, not
    from the normal.

    returns
        mu_next: float    one-step-ahead conditional mean (return units)
        sigma_next: float one-step-ahead conditional vol (return units)
        z: np.ndarray     standardised residuals (zero mean, ~unit variance)
    """
    from arch import arch_model

    scaled = r * 100  # percentage space for optimiser stability
    am = arch_model(scaled, mean="Constant", vol="Garch",
                    p=p, o=o, q=q, dist="normal", rescale=False)
    res = am.fit(disp="off")

    if res.convergence_flag != 0:
        raise RuntimeError("GARCH fit did not converge; FHS output unreliable.")

    fc = res.forecast(horizon=1, reindex=False)
    mu_next = float(fc.mean.values[-1, 0]) / 100
    sigma_next = float(np.sqrt(fc.variance.values[-1, 0])) / 100

    # std_resid = (r_t - mu) / sigma_t, in-sample
    z = res.std_resid.to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    return mu_next, sigma_next, z


def var_fhs(
    r: pd.Series,
    alpha: float = 0.99,
    o: int = 0,
) -> float:
    """
    1-day Filtered Historical Simulation VaR (GARCH-filtered bootstrap).

    Steps
    -----
    1. Fit GARCH(1,o,1) on the window -> conditional vol sigma_t
    2. Standardise: z_t = (r_t - mu) / sigma_t
    3. Forecast sigma_{T+1|T}
    4. Build scenarios: r_i = mu + sigma_{T+1|T} * z_i
    5. VaR = -(1-alpha) empirical quantile of scenarios

    Unlike var_garch, tail shape comes from the data (fat tails, skew survive).
    Unlike var_historical, scenarios are rescaled to today's volatility regime.

    parameters
        r: pd.Series
            historical return window (500+ obs recommended)
        alpha: float
            confidence level
        o: int
            0 = symmetric GARCH, 1 = GJR-GARCH

    returns
        float
            positive VaR value (loss convention)
    """
    mu_next, sigma_next, z = _fit_garch_fhs(r, o=o)
    scenarios = mu_next + sigma_next * z
    return float(-np.quantile(scenarios, 1 - alpha))


def cvar_fhs(
    r: pd.Series,
    alpha: float = 0.99,
    o: int = 0,
) -> float:
    """
    1-day FHS expected shortfall: mean of filtered scenarios beyond the FHS VaR.
    """
    mu_next, sigma_next, z = _fit_garch_fhs(r, o=o)
    scenarios = mu_next + sigma_next * z
    cutoff = np.quantile(scenarios, 1 - alpha)
    return float(-scenarios[scenarios <= cutoff].mean())

def _gpd_tail_var_es(
    losses: np.ndarray,
    alpha: float,
    threshold_quantile: float,
    min_exceedances: int = 20,
) -> tuple[float, float, float, float]:
    """
    Fit a GPD to the upper tail of `losses` (POT) and return the VaR and ES
    at level alpha, in the same units as `losses`.

    returns
        var, es, xi, beta
    """
    if threshold_quantile >= alpha:
        raise ValueError("threshold_quantile must be below alpha.")

    u = float(np.quantile(losses, threshold_quantile))
    exceedances = losses[losses > u] - u
    n, n_u = len(losses), len(exceedances)

    if n_u < min_exceedances:
        raise ValueError(f"Only {n_u} exceedances (< {min_exceedances}); lower threshold_quantile or use a longer window.")

    xi, _, beta = genpareto.fit(exceedances, floc=0)

    if xi >= 1:
        raise ValueError(f"Fitted xi={xi:.3f} >= 1: infinite-mean tail, ES undefined.")

    p = (1 - alpha) * n / n_u  # = (1-alpha) / P(L > u)

    if abs(xi) < 1e-8:
        var = u + beta * np.log(1.0 / p)
    else:
        var = u + (beta / xi) * (p ** (-xi) - 1.0)

    # GPD expected shortfall (McNeil & Frey 2000), valid for xi < 1
    es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
    return float(var), float(es), float(xi), float(beta)


def fhs_evt_risk(
    r: pd.Series,
    alpha: float = 0.99,
    threshold_quantile: float = 0.90,
    o: int = 0,
    return_diagnostics: bool = False,
):
    """
    1-day FHS-EVT VaR and ES (McNeil & Frey, 2000).

    Steps
    -----
    1. Fit GARCH(1,o,1) -> standardised residuals z_t, forecasts mu, sigma_{T+1|T}
    2. Standardised losses: l_t = -z_t
    3. Fit GPD to exceedances of l_t over u (POT on the residuals, NOT the raw returns)
    4. VaR_z, ES_z from the GPD tail
    5. Rescale: VaR = -mu_next + sigma_next * VaR_z
                ES  = -mu_next + sigma_next * ES_z

    The tail is extrapolated by the GPD instead of capped at the largest observed
    residual, so it stays valid at 99.5%/99.9%, where plain FHS runs out of data.

    parameters
        r: pd.Series
            historical return window (750+ obs recommended)
        alpha: float
            confidence level (must exceed threshold_quantile)
        threshold_quantile: float
            quantile of standardised losses used as GPD threshold. Default 0.90
            (not 0.95): z is already filtered, so you can afford a lower
            threshold, and you need the exceedances.
        o: int
            0 = GARCH, 1 = GJR-GARCH
        return_diagnostics: bool
            if True, also return dict with xi, beta, sigma_next, mu_next

    returns
        (var, es) as positive losses, plus diagnostics dict if requested
    """
    mu_next, sigma_next, z = _fit_garch_fhs(r, o=o)
    var_z, es_z, xi, beta = _gpd_tail_var_es(-z, alpha, threshold_quantile)

    var = -mu_next + sigma_next * var_z
    es = -mu_next + sigma_next * es_z

    if return_diagnostics:
        return float(var), float(es), {"xi": xi, "beta": beta,
                                       "sigma_next": sigma_next, "mu_next": mu_next}
    return float(var), float(es)


def var_fhs_evt(r: pd.Series, alpha: float = 0.99,
                threshold_quantile: float = 0.90, o: int = 0) -> float:
    return fhs_evt_risk(r, alpha, threshold_quantile, o)[0]


def cvar_fhs_evt(r: pd.Series, alpha: float = 0.99,
                 threshold_quantile: float = 0.90, o: int = 0) -> float:
    return fhs_evt_risk(r, alpha, threshold_quantile, o)[1]

