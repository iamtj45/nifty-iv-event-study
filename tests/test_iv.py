"""Tests for the Black-Scholes pricer and the implied-vol solver.

Run with:  pytest

These are the guard rails for iv.py. If you rewrite the maths in bs_price or
the bounds in price_bounds, these tests are what tell you whether you got it
right -- a sign error in the put formula breaks parity, a wrong bound lets an
impossible price return a number instead of NaN.
"""

import math
import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iv import CALL, PUT, VOL_MAX, VOL_MIN, bs_price, implied_vol, price_bounds

# A plausible Nifty setup: at the money, a month out, at the project's rate.
S = 24000.0
K = 24000.0
T = 30 / 365
R = 0.065

VOLS = [0.05, 0.10, 0.15, 0.30, 0.50, 1.00, 1.50]
KINDS = [CALL, PUT]


# --------------------------------------------------------------------------
# The solver recovers a volatility we chose ourselves
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", VOLS)
@pytest.mark.parametrize("kind", KINDS)
def test_round_trip_recovers_known_vol(sigma, kind):
    """Price at a known sigma, solve for it, get it back."""
    price = bs_price(S, K, T, R, sigma, kind)
    assert price > 0
    assert implied_vol(price, S, K, T, R, kind) == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("moneyness", [0.90, 0.95, 1.00, 1.05, 1.10])
@pytest.mark.parametrize("kind", KINDS)
def test_round_trip_away_from_the_money(moneyness, kind):
    """Still recovers sigma when the strike is not at the money.

    This is where a too-narrow bracket or a one-sided sign error shows up:
    the at-the-money case can pass while the wings fail.
    """
    strike = S * moneyness
    sigma = 0.15
    price = bs_price(S, strike, T, R, sigma, kind)
    assert implied_vol(price, S, strike, T, R, kind) == pytest.approx(sigma, abs=1e-6)


# --------------------------------------------------------------------------
# Put-call parity: catches sign errors in the put formula
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", VOLS)
@pytest.mark.parametrize("moneyness", [0.90, 1.00, 1.10])
def test_put_call_parity(sigma, moneyness):
    """Call - Put must equal S - K*exp(-rT), exactly.

    This is an identity, not an approximation: it follows from the payoffs
    alone and holds for any volatility. If it fails, a sign in bs_price is
    wrong -- and a wrong sign gives put IVs that look almost plausible, which
    is far more dangerous than an obvious crash.
    """
    strike = S * moneyness
    call = bs_price(S, strike, T, R, sigma, CALL)
    put = bs_price(S, strike, T, R, sigma, PUT)
    forward = S - strike * math.exp(-R * T)
    assert call - put == pytest.approx(forward, abs=1e-8)


# --------------------------------------------------------------------------
# Impossible prices return NaN rather than a wrong number
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind,strike", [(CALL, S * 0.70), (PUT, S * 1.30)])
def test_price_below_intrinsic_returns_nan(kind, strike):
    """No volatility can produce a price under intrinsic value.

    The strike has to be in the money for the test to mean anything. An
    at-the-money put has *zero* intrinsic value when rates are positive --
    its lower bound is max(0, K*exp(-rT) - S) and the forward sits above
    spot -- so there would be no price below intrinsic to try.
    """
    low, _ = price_bounds(S, strike, T, R, kind)
    assert low > 1.0, "test needs a strike with real intrinsic value"
    assert math.isnan(implied_vol(low - 1.0, S, strike, T, R, kind))
    assert math.isnan(implied_vol(low * 0.5, S, strike, T, R, kind))


@pytest.mark.parametrize("kind", KINDS)
def test_price_above_maximum_returns_nan(kind):
    """Nor one above what infinite volatility would give."""
    _, high = price_bounds(S, K, T, R, kind)
    assert math.isnan(implied_vol(high + 1.0, S, K, T, R, kind))


@pytest.mark.parametrize("kind", KINDS)
def test_zero_and_negative_time_return_nan(kind):
    """T sits in a denominator, so expiry day must not divide by zero."""
    assert math.isnan(implied_vol(100.0, S, K, 0.0, R, kind))
    assert math.isnan(implied_vol(100.0, S, K, -1.0, R, kind))
    assert math.isnan(bs_price(S, K, 0.0, R, 0.15, kind))


@pytest.mark.parametrize("kind", KINDS)
def test_nonsense_inputs_return_nan(kind):
    """Zero or negative prices, spots and strikes are data errors, not crashes."""
    assert math.isnan(implied_vol(0.0, S, K, T, R, kind))
    assert math.isnan(implied_vol(-5.0, S, K, T, R, kind))
    assert math.isnan(implied_vol(100.0, 0.0, K, T, R, kind))
    assert math.isnan(implied_vol(100.0, S, 0.0, T, R, kind))
    assert math.isnan(bs_price(S, K, T, R, 0.0, kind))


def test_unknown_kind_raises():
    """Bad data returns NaN; a typo in the calling code should be loud."""
    with pytest.raises(ValueError):
        bs_price(S, K, T, R, 0.15, "XX")
    with pytest.raises(ValueError):
        price_bounds(S, K, T, R, "XX")


# --------------------------------------------------------------------------
# The property the solver depends on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_price_strictly_increasing_in_vol(kind):
    """Price must rise with volatility, everywhere.

    Bracketing only works because of this. If it failed, two different
    volatilities could give the same price and the solved IV would be
    meaningless.
    """
    prices = [bs_price(S, K, T, R, v, kind) for v in VOLS]
    assert all(b > a for a, b in zip(prices, prices[1:]))


@pytest.mark.parametrize("kind", KINDS)
def test_bounds_are_the_limits_of_the_pricer(kind):
    """price_bounds must agree with what bs_price actually does at the edges."""
    low, high = price_bounds(S, K, T, R, kind)
    assert bs_price(S, K, T, R, VOL_MIN, kind) == pytest.approx(low, abs=0.01)
    assert low <= bs_price(S, K, T, R, VOL_MAX, kind) <= high + 1e-6


# --------------------------------------------------------------------------
# Against real NSE prices (skipped if the data is not downloaded)
# --------------------------------------------------------------------------

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "raw")


@pytest.mark.skipif(not glob.glob(os.path.join(RAW, "*.csv.zip")),
                    reason="no bhavcopies downloaded; run download.py first")
def test_solves_real_nse_closes():
    """Most traded near-the-money options on a real day must solve sensibly.

    Not every close is usable -- stale prices below intrinsic are normal and
    correctly return NaN. What matters is that the bulk solve, and that the
    answers land in a believable band for Nifty.
    """
    import csv
    import datetime as dt
    import io
    import zipfile

    newest = sorted(glob.glob(os.path.join(RAW, "*.csv.zip")))[-1]
    day = dt.datetime.strptime(os.path.basename(newest)[:8], "%Y%m%d").date()

    with zipfile.ZipFile(newest) as z:
        rows = list(csv.DictReader(io.StringIO(z.read(z.namelist()[0]).decode())))

    assert "FinInstrmTp" in rows[0], "expected the post-2024 bhavcopy layout"
    opts = [r for r in rows
            if r["TckrSymb"] == "NIFTY" and r["FinInstrmTp"] == "IDO"
            and float(r["TtlTradgVol"] or 0) > 0]
    assert opts, "no traded NIFTY options found"

    spot = float(opts[0]["UndrlygPric"])
    expiry = sorted({dt.date.fromisoformat(r["XpryDt"]) for r in opts
                     if (dt.date.fromisoformat(r["XpryDt"]) - day).days >= 7})[0]
    years = (expiry - day).days / 365

    # Within 5% of spot: the strikes the study actually uses.
    near = [r for r in opts
            if dt.date.fromisoformat(r["XpryDt"]) == expiry
            and abs(float(r["StrkPric"]) - spot) / spot <= 0.05]
    assert len(near) >= 10, f"only {len(near)} near-the-money contracts"

    solved = [v for v in (implied_vol(float(r["ClsPric"]), spot,
                                      float(r["StrkPric"]), years, R, r["OptnTp"])
                          for r in near)
              if not math.isnan(v)]

    assert len(solved) / len(near) > 0.70, (
        f"only {len(solved)}/{len(near)} solved on {day}")
    assert all(0.01 < v < 2.00 for v in solved), "implausible IV for Nifty"

    median = sorted(solved)[len(solved) // 2]
    assert 0.05 < median < 0.60, f"median IV {median:.1%} on {day} looks wrong"
