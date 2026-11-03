"""Check the C++ solver against the Python one, and time both.

    python bench.py --build        compile cpp/iv.cpp first, then run
    python bench.py               use the library already built

Two questions, and the second one is only worth asking once the first is
answered:

    Do they agree?   Two independent implementations matching to twelve
                     decimal places is a far stronger statement about the
                     maths than one implementation passing its own tests.
    How much faster? Measured, not assumed -- and measured honestly, which
                     turns out to matter a lot.

On the honest part. Calling a C function once per option through ctypes costs
roughly as much in marshalling as the arithmetic saves, so a per-call
benchmark mostly measures ctypes. The library therefore has a batch entry
point that crosses the boundary once for a whole array, and this script times
all three -- Python, C++ per call, C++ batched -- so the overhead is visible
rather than hidden in whichever number flatters the result.

Two test sets, because they fail differently:

    a synthetic grid   spans moneyness and maturity evenly, so a systematic
                       difference in one corner shows up clearly
    one real bhavcopy  thousands of genuine NSE closes, including the stale
                       and out-of-bounds ones that the clean grid never
                       produces -- the cases where the two could disagree
                       about whether an answer exists at all
"""

import argparse
import ctypes
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
CPP = BASE / "cpp"

# Windows builds a .dll, everything else a .so. Nothing else differs: the
# source has one #if for the export attribute and no other platform code.
LIB_NAME = "libiv.dll" if sys.platform == "win32" else "libiv.so"
LIB_PATH = CPP / LIB_NAME

BUILD_WIN = ["g++", "-O2", "-shared", "-static-libgcc", "-static-libstdc++",
             "-o", LIB_NAME, "iv.cpp"]
BUILD_UNIX = ["g++", "-O2", "-shared", "-fPIC", "-o", LIB_NAME, "iv.cpp"]

RATE = 0.065
KIND_CALL, KIND_PUT = 0, 1

# Below this much price movement per volatility point, the implied
# volatility is not recoverable from the price in double precision.
VEGA_FLOOR = 1e-6


def build():
    """Compile cpp/iv.cpp into the shared library."""
    cmd = BUILD_WIN if sys.platform == "win32" else BUILD_UNIX
    print(f"  {' '.join(cmd)}")
    done = subprocess.run(cmd, cwd=CPP, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout)
        print(done.stderr, file=sys.stderr)
        sys.exit("compile failed")
    size = LIB_PATH.stat().st_size
    print(f"  built {LIB_NAME} ({size / 1024:.0f} KB)\n")


def load():
    """Open the library and declare the signatures ctypes needs."""
    if not LIB_PATH.exists():
        sys.exit(f"{LIB_PATH} is missing -- run `python bench.py --build`")

    lib = ctypes.CDLL(str(LIB_PATH))

    d, i = ctypes.c_double, ctypes.c_int
    pd_, pi = ctypes.POINTER(d), ctypes.POINTER(i)

    lib.iv_bs_price.argtypes = [d, d, d, d, d, i]
    lib.iv_bs_price.restype = d

    lib.iv_implied_vol.argtypes = [d, d, d, d, d, i]
    lib.iv_implied_vol.restype = d

    lib.iv_implied_vol_batch.argtypes = [pd_, pd_, pd_, pd_, pd_, pi, pd_, i]
    lib.iv_implied_vol_batch.restype = None

    lib.iv_version.restype = ctypes.c_char_p
    return lib


def cpp_batch(lib, price, S, K, T, r, kind):
    """One call for the whole array."""
    n = len(price)
    out = np.empty(n, dtype=np.float64)

    def dp(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.iv_implied_vol_batch(
        dp(price), dp(S), dp(K), dp(T), dp(r),
        kind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dp(out), n)
    return out


def python_all(price, S, K, T, r, kind):
    from iv import CALL, PUT, implied_vol
    kinds = {KIND_CALL: CALL, KIND_PUT: PUT}
    return np.array([
        implied_vol(price[i], S[i], K[i], T[i], r[i], kinds[int(kind[i])])
        for i in range(len(price))
    ])


# ---------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------

def synthetic_set():
    """An even grid: price each option at a known vol, then solve it back.

    Built by pricing rather than by inventing prices, so every case has an
    answer that both solvers must find. A difference here is a difference in
    the root-finding, not a disagreement about whether a root exists.
    """
    from iv import CALL, PUT, bs_price

    spot = 24000.0
    rows = []
    for moneyness in np.linspace(0.85, 1.15, 25):          # deep ITM to deep OTM
        for days in (7, 14, 30, 60, 90):
            for sigma in (0.08, 0.12, 0.20, 0.35, 0.60):
                for kind, py_kind in ((KIND_CALL, CALL), (KIND_PUT, PUT)):
                    strike = round(spot * moneyness / 50) * 50
                    years = days / 365
                    price = bs_price(spot, strike, years, RATE, sigma, py_kind)
                    if price is None or not np.isfinite(price) or price <= 0.01:
                        continue
                    rows.append((price, spot, strike, years, RATE, kind))

    a = np.array(rows, dtype=np.float64)
    return (a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(),
            a[:, 3].copy(), a[:, 4].copy(),
            a[:, 5].astype(np.int32).copy())


def real_set(limit=6000):
    """Every traded option from one real bhavcopy.

    This is the set that matters. Real closes include stale prices, prices
    below intrinsic value and prices above the no-arbitrage maximum -- cases
    where the right answer is "there is no implied volatility". Both solvers
    must agree on which those are, not just on the numbers.
    """
    from build import RAW, read_day
    from iv import CALL

    paths = sorted(RAW.glob("*.csv.zip"))
    if not paths:
        return None
    path = paths[len(paths) // 2]                 # mid-period, nothing special

    rows, underlying = read_day(path)
    if not rows or not underlying:
        return None
    day = dt.datetime.strptime(path.stem[:8], "%Y%m%d").date()

    out = []
    for expiry, strike, kind, close, volume in rows:
        years = (expiry - day).days / 365
        if volume <= 0 or years <= 0 or close <= 0:
            continue
        out.append((close, underlying, strike, years, RATE,
                    KIND_CALL if kind == CALL else KIND_PUT))
        if len(out) >= limit:
            break

    if not out:
        return None
    a = np.array(out, dtype=np.float64)
    print(f"  from {path.name}, spot {underlying:,.2f}, {len(out):,} contracts")
    return (a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(),
            a[:, 3].copy(), a[:, 4].copy(),
            a[:, 5].astype(np.int32).copy())


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(name, lib, data, time_percall):
    from iv import CALL, PUT, bs_price

    price, S, K, T, r, kind = data
    n = len(price)
    print(f"\n{name}  ({n:,} options)")

    t0 = time.perf_counter()
    py = python_all(price, S, K, T, r, kind)
    py_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    cp = cpp_batch(lib, price, S, K, T, r, kind)
    cpp_s = time.perf_counter() - t0

    percall_s = None
    if time_percall:
        t0 = time.perf_counter()
        for i in range(n):
            lib.iv_implied_vol(price[i], S[i], K[i], T[i], r[i], int(kind[i]))
        percall_s = time.perf_counter() - t0

    # Agreement. Both NaN counts as agreement -- "no solution" is an answer,
    # and the two must reach it on the same inputs.
    py_nan, cp_nan = np.isnan(py), np.isnan(cp)
    both_ok = ~py_nan & ~cp_nan
    disagree_nan = int((py_nan != cp_nan).sum())

    # Deep in-the-money options are almost entirely intrinsic value, so their
    # price barely responds to volatility at all. Where vega is near zero the
    # implied volatility is not determined by the price at double precision:
    # thousands of different sigmas reproduce the same price exactly, and two
    # correct solvers can legitimately stop at different ones.
    #
    # Those cases are counted and reported, not quietly tolerated by widening
    # the threshold -- a test loosened until it passes has stopped testing
    # anything. Agreement is judged only where the root actually exists to
    # find. Real market data barely produces these: a contract this far in the
    # money usually has no volume, or a price outside the no-arbitrage bounds,
    # and is rejected before the solver sees it.
    vega = np.full(n, np.nan)
    ok = np.where(both_ok)[0]
    for i in ok:
        kind_py = CALL if int(kind[i]) == KIND_CALL else PUT
        up = bs_price(S[i], K[i], T[i], r[i], py[i] + 0.005, kind_py)
        dn = bs_price(S[i], K[i], T[i], r[i], py[i] - 0.005, kind_py)
        if up is not None and dn is not None:
            vega[i] = abs(up - dn)          # price move per 1 vol point

    determined = both_ok & (vega > VEGA_FLOOR)
    flat = both_ok & ~(vega > VEGA_FLOOR)
    diff = np.abs(py - cp)
    worst = float(np.max(diff[determined])) if determined.any() else 0.0

    print(f"  solved by both        {int(both_ok.sum()):,} / {n:,}")
    print(f"  unsolvable in both    {int((py_nan & cp_nan).sum()):,}")
    print(f"  NaN disagreements     {disagree_nan}"
          + ("" if disagree_nan == 0 else "   <-- INVESTIGATE"))
    print(f"  root not determined   {int(flat.sum()):,}"
          f"   (vega below {VEGA_FLOOR:.0e} per vol point: "
          f"any sigma reproduces the price)")
    print(f"  largest difference    {worst:.3e}"
          f"   over the {int(determined.sum()):,} with a findable root")
    if flat.any():
        print(f"    of those flat cases, largest difference {np.max(diff[flat]):.3e}"
              f" -- expected, not a disagreement about the maths")

    print(f"  python                {py_s * 1000:9.1f} ms"
          f"   ({py_s / n * 1e6:6.2f} us each)")
    print(f"  c++ batched           {cpp_s * 1000:9.1f} ms"
          f"   ({cpp_s / n * 1e6:6.2f} us each)   {py_s / cpp_s:5.1f}x faster")
    if percall_s:
        print(f"  c++ one call at a time{percall_s * 1000:9.1f} ms"
              f"   ({percall_s / n * 1e6:6.2f} us each)   "
              f"{py_s / percall_s:5.1f}x faster")
        print(f"  ctypes overhead       {(percall_s - cpp_s) / n * 1e6:6.2f} us "
              f"per call -- the cost of crossing the boundary, which is why "
              f"the batch entry point exists")

    return worst, disagree_nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="compile first")
    ap.add_argument("--no-percall", action="store_true",
                    help="skip the per-call timing")
    args = ap.parse_args()

    if args.build:
        build()

    lib = load()
    print(f"loaded {LIB_PATH.name}: {lib.iv_version().decode()}")

    # A single price both ways, before anything else. If this disagrees there
    # is no point measuring the solver.
    from iv import CALL, bs_price
    py_p = bs_price(24000.0, 24000.0, 30 / 365, RATE, 0.15, CALL)
    cp_p = lib.iv_bs_price(24000.0, 24000.0, 30 / 365, RATE, 0.15, KIND_CALL)
    print(f"  bs_price check: python {py_p:.10f}  c++ {cp_p:.10f}  "
          f"diff {abs(py_p - cp_p):.2e}")

    worst_all, bad = 0.0, 0

    w, d = compare("synthetic grid", lib, synthetic_set(), not args.no_percall)
    worst_all, bad = max(worst_all, w), bad + d

    real = real_set()
    if real is None:
        print("\nreal bhavcopy      skipped (no data/raw)")
    else:
        w, d = compare("one real bhavcopy", lib, real, not args.no_percall)
        worst_all, bad = max(worst_all, w), bad + d

    print(f"\n  worst difference anywhere: {worst_all:.3e}")
    if bad or worst_all > 1e-6:
        print("  the two implementations do not agree -- do not trust either")
        sys.exit(1)
    print("  the two implementations agree")


if __name__ == "__main__":
    main()
