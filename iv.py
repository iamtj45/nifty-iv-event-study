"""Black-Scholes pricing and implied volatility for Nifty index options.

Two public functions:

    bs_price(S, K, T, r, sigma, kind)      -> option price
    implied_vol(price, S, K, T, r, kind)   -> volatility, or NaN

Conventions, fixed here so nothing downstream has to guess:

    S       Nifty level (spot)
    K       strike
    T       years to expiry, as calendar days / 365
    r       annual risk-free rate as a decimal (0.065)
    sigma   annual volatility as a decimal (0.15 means 15%)
    kind    "CE" for a call or "PE" for a put, matching NSE's own column

No dividend yield. Nifty is a price index, so its forward sits slightly below
spot by the dividends expected before expiry, and ignoring that biases IV a
little -- puts marginally high, calls marginally low. The standard fix is
Black-76 on the near-month future instead of spot; data/raw keeps the futures
rows if we ever want it.

This module holds no date arithmetic and reads no files, which keeps it easy to
test and a direct template for the C++ port in cpp/iv.cpp.
"""

import math

from scipy.optimize import brentq

# NSE's option-type codes, used as-is so no translation layer is needed.
CALL = "CE"
PUT = "PE"

# Search bracket for the solver: 0.01% to 500% annualised volatility.
# Wide enough for anything real; a true IV outside it returns NaN.
VOL_MIN = 1e-4
VOL_MAX = 5.0

NAN = float("nan")


def norm_cdf(x):
    """Standard normal CDF.

    Uses erfc rather than scipy.stats.norm.cdf so the C++ port can use
    std::erfc and match this to the last decimal place.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_price(S, K, T, r, sigma, kind):
    """Black-Scholes price of a European call ("CE") or put ("PE").

    Returns NaN where the formula is undefined -- zero or negative spot,
    strike, time or volatility. Nifty options are European, so this is the
    right model for them.

    An unrecognised `kind` raises: bad data should return NaN, but a typo in
    the calling code should be loud.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return NAN

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pv_strike = K * math.exp(-r * T)

    if kind == CALL:
        return S * norm_cdf(d1) - pv_strike * norm_cdf(d2)
    if kind == PUT:
        return pv_strike * norm_cdf(-d2) - S * norm_cdf(-d1)
    raise ValueError(f"kind must be {CALL!r} or {PUT!r}, got {kind!r}")


def price_bounds(S, K, T, r, kind):
    """The (low, high) prices any volatility can produce.

    As sigma falls to zero an option approaches its intrinsic value; as sigma
    grows a call approaches S and a put approaches the present value of the
    strike. A market price outside this band cannot be inverted at all, which
    is the single most common reason a real NSE close is unusable.
    """
    pv_strike = K * math.exp(-r * T)
    if kind == CALL:
        return max(0.0, S - pv_strike), S
    if kind == PUT:
        return max(0.0, pv_strike - S), pv_strike
    raise ValueError(f"kind must be {CALL!r} or {PUT!r}, got {kind!r}")


def implied_vol(price, S, K, T, r, kind):
    """The volatility that makes bs_price equal `price`, or NaN if none does.

    Never raises on bad data: build.py calls this hundreds of thousands of
    times and one exception would end the run. Every failure returns NaN and
    is counted as unsolved in the quality report.

    Brent's method is used rather than Newton's. Brent is guaranteed to
    converge once the root is bracketed, while Newton relies on the derivative
    (vega) and can step outside the valid range for options far from the money
    or close to expiry -- exactly the cases this data is full of.
    """
    if not (price > 0) or S <= 0 or K <= 0 or T <= 0:
        return NAN

    low, high = price_bounds(S, K, T, r, kind)
    if price < low or price > high:
        return NAN                     # below intrinsic, or above the maximum

    def excess(sigma):
        """How much bs_price overshoots the market price at this sigma."""
        return bs_price(S, K, T, r, sigma, kind) - price

    # The price is monotonic in sigma, so a root exists only if the ends of
    # the bracket straddle zero. Checking here means brentq is never handed a
    # bracket it would reject.
    at_min, at_max = excess(VOL_MIN), excess(VOL_MAX)
    if math.isnan(at_min) or math.isnan(at_max) or at_min * at_max > 0:
        return NAN

    try:
        return brentq(excess, VOL_MIN, VOL_MAX, xtol=1e-8, maxiter=100)
    except (ValueError, RuntimeError):
        return NAN


if __name__ == "__main__":
    # Quick self-check: price at a known volatility, then recover it.
    S, K, T, r = 24000.0, 24000.0, 30 / 365, 0.065

    print(f"S={S:,.0f}  K={K:,.0f}  T={T:.4f}y  r={r:.1%}\n")
    for sigma in (0.05, 0.15, 0.50, 1.50):
        for kind in (CALL, PUT):
            p = bs_price(S, K, T, r, sigma, kind)
            back = implied_vol(p, S, K, T, r, kind)
            print(f"  sigma={sigma:>5.2f}  {kind}  price={p:>10,.2f}  "
                  f"recovered={back:.6f}  error={abs(back - sigma):.2e}")

    call = bs_price(S, K, T, r, 0.15, CALL)
    put = bs_price(S, K, T, r, 0.15, PUT)
    parity = (call - put) - (S - K * math.exp(-r * T))
    print(f"\n  put-call parity residual: {parity:.2e}")

    low, _ = price_bounds(S, K, T, r, CALL)
    print(f"  price below intrinsic ({low:.2f}): "
          f"{implied_vol(max(low - 1.0, 0.01), S, K, T, r, CALL)}")
    print(f"  T=0: {implied_vol(100.0, S, K, 0.0, r, CALL)}")
