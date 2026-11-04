"""Turn the downloaded bhavcopies into daily ATM implied volatility and the
event study.

    python build.py

Reads   data/raw/*.csv.zip, data/spot.csv, data/vix.csv, data/events.csv
Writes  data/daily_iv.csv   one row per trading day
        data/quality.csv    one row per trading day: what was usable
        data/results.csv    one row per event
        data/curves.csv     average IV path per event type, offset -10..+10
        data/summary.csv    one row per event type, with confidence ranges
        data/placebo.csv    one row per event type: is the effect real?

The measurement, in six steps:

  1. For every trading day, pick the nearest expiry at least 7 calendar days
     out, then the strike closest to spot that has a traded call *and* a
     traded put.
  2. Solve implied volatility for both and average them. That is the day's
     ATM IV. Repeat for the following expiry, so the gap between them can be
     measured, and for strikes about 5% either side, to measure skew.
  3. For every event, record ATM IV on the trading day before, ATM IV at the
     close on the event day, and the percentage change between them.
  4. Measure what Nifty actually did over the 10 trading days after, as
     annualised realised volatility, and compare it with the IV that was
     being charged beforehand. Separately compare the move the straddle
     implied with the move that actually happened by the event-day close.
  5. Average the IV path, the next-expiry path and skew per event type over
     a window of 10 trading days either side.
  6. Ask whether any of it is unusual: rank each event type's average change
     against groups of ordinary days of the same size, and put a confidence
     range on every average.

The statistics in step 6 live in analysis.py, which knows nothing about files
or dates. This module prepares the inputs and writes the outputs.
"""

import argparse
import csv
import datetime as dt
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (NORMAL_DAY_GAP, bootstrap_ci, expected_move_one_day,
                      expected_move_pct, normal_day_mask, one_day_changes,
                      placebo_percentile, skew)
from iv import CALL, PUT, implied_vol

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"

RATE = 0.065              # annual, flat across the period (see README limits)
MIN_DAYS_TO_EXPIRY = 7    # calendar days; IV goes unstable into expiry
WINDOW = 10               # trading days either side of an event
SYMBOL = "NIFTY"

# Strikes used for skew: a put this far below spot against a call this far
# above. Five percent is far enough to be clearly out of the money, close
# enough that both still trade on an ordinary day.
SKEW_OFFSET = 0.05

# ---------------------------------------------------------------------------
# Decision 1: where spot comes from.
#
# The post-2024 bhavcopy carries UndrlygPric, the Nifty level recorded at the
# same moment as the option close. Using it makes the two simultaneous, which
# the older files cannot do -- there we fall back to Yahoo's daily close, which
# is stamped at a slightly different instant. It also supplies a spot price for
# the Sunday Budget of 1 Feb 2026, a day Yahoo does not publish at all and one
# of only five Budget events in the study.
#
# Set to False to use Yahoo's close everywhere instead.
# ---------------------------------------------------------------------------
PREFER_UNDERLYING_FROM_BHAVCOPY = True

# ---------------------------------------------------------------------------
# Decision 2: four weekend sessions that are not ordinary trading days.
#
# The Diwali Muhurat session is a ceremonial hour; the three Saturdays are
# NSE's disaster-recovery drills, run live from the backup site. All four have
# thin, artificial volume, and an IV taken from them is not comparable with a
# normal day's. They are dropped from the daily series and from the trading-day
# calendar, and listed in quality.csv so the exclusion is visible.
#
# The two weekend *Budget* sessions are full 9:15-3:30 days and are kept: they
# are event days, and two of only five Budgets.
# ---------------------------------------------------------------------------
THIN_SESSIONS = {
    dt.date(2023, 11, 12),   # Diwali Muhurat trading, Sunday
    dt.date(2024, 1, 20),    # NSE special session, Saturday
    dt.date(2024, 3, 2),     # NSE special session, Saturday
    dt.date(2024, 5, 18),    # NSE special session, Saturday
}


# ---------------------------------------------------------------------------
# Reading one day's bhavcopy
# ---------------------------------------------------------------------------

def parse_expiry(text):
    """Expiry date from either layout.

    New files use ISO ("2026-11-23"). Old files use "23-Nov-2026", and the
    same file writes TIMESTAMP in upper case, so parsing is case-insensitive.
    """
    text = text.strip()
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return dt.datetime.strptime(text.title(), "%d-%b-%Y").date()


def read_day(path, symbol=SYMBOL):
    """Return (rows, underlying) for one bhavcopy.

    Each row is (expiry, strike, kind, close, volume). `underlying` is NSE's
    own spot print, or None for the old layout which does not carry it.

    `symbol` selects the index: NIFTY or BANKNIFTY. download.py kept both, so
    the same 971 files serve either study.
    """
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", "replace")

    reader = csv.DictReader(io.StringIO(text))
    fields = [f for f in (reader.fieldnames or []) if f]
    new = "FinInstrmTp" in fields

    rows = []
    underlying = None
    for r in reader:
        if new:
            if r["FinInstrmTp"] != "IDO" or r["TckrSymb"] != symbol:
                continue
            expiry, strike, kind = r["XpryDt"], r["StrkPric"], r["OptnTp"]
            close, volume = r["ClsPric"], r["TtlTradgVol"]
            if underlying is None and r.get("UndrlygPric"):
                underlying = float(r["UndrlygPric"])
        else:
            if r["INSTRUMENT"] != "OPTIDX" or r["SYMBOL"] != symbol:
                continue
            expiry, strike, kind = r["EXPIRY_DT"], r["STRIKE_PR"], r["OPTION_TYP"]
            close, volume = r["CLOSE"], r["CONTRACTS"]

        if kind not in (CALL, PUT):
            continue
        try:
            rows.append((parse_expiry(expiry), float(strike), kind,
                         float(close), float(volume or 0)))
        except ValueError:
            continue

    return rows, underlying


# ---------------------------------------------------------------------------
# Solving one strike, and one at-the-money pair
# ---------------------------------------------------------------------------

def best_quote(chain, strike, kind):
    """The most-traded close for one strike and option type, or None.

    A strike should appear once per type per expiry, but taking the highest
    volume costs nothing and guards against a duplicated row.
    """
    matches = [r for r in chain if r[1] == strike and r[2] == kind]
    return max(matches, key=lambda r: r[4]) if matches else None


def solve_near_strike(chain, target, kind, spot, years):
    """IV of the traded strike closest to `target`, for one option type.

    Used for the skew legs, where we want a strike about 5% away from spot
    rather than the one nearest to it.
    """
    strikes = {r[1] for r in chain if r[2] == kind}
    if not strikes:
        return np.nan, np.nan
    strike = min(strikes, key=lambda k: abs(k - target))
    quote = best_quote(chain, strike, kind)
    if quote is None:
        return np.nan, np.nan
    return implied_vol(quote[3], spot, strike, years, RATE, kind), strike


def solve_atm_pair(chain, spot, day, expiry):
    """ATM IV for one expiry: the strike nearest spot with both legs traded.

    Requiring a traded call *and* a traded put matters -- averaging one leg
    against nothing would half-measure the day, and a strike can easily have
    volume on one side only.
    """
    calls = {r[1] for r in chain if r[2] == CALL}
    puts = {r[1] for r in chain if r[2] == PUT}
    both = calls & puts
    if not both:
        return None

    strike = min(both, key=lambda k: abs(k - spot))
    years = (expiry - day).days / 365
    call, put = best_quote(chain, strike, CALL), best_quote(chain, strike, PUT)
    if call is None or put is None:
        return None

    call_iv = implied_vol(call[3], spot, strike, years, RATE, CALL)
    put_iv = implied_vol(put[3], spot, strike, years, RATE, PUT)
    if np.isnan(call_iv) and np.isnan(put_iv):
        return None

    return {
        "strike": strike,
        "expiry": expiry,
        "days_to_expiry": (expiry - day).days,
        "years": years,
        "call_price": call[3],
        "put_price": put[3],
        "call_iv": call_iv * 100 if not np.isnan(call_iv) else np.nan,
        "put_iv": put_iv * 100 if not np.isnan(put_iv) else np.nan,
        "atm_iv": float(np.nanmean([call_iv, put_iv])) * 100,
        "unsolved": int(np.isnan(call_iv)) + int(np.isnan(put_iv)),
    }


# ---------------------------------------------------------------------------
# One day
# ---------------------------------------------------------------------------

def atm_for_day(day, rows, spot):
    """Everything measured from one day's option chain.

    Returns (record_or_None, quality_dict).
    """
    quality = {
        "date": day,
        "options": len(rows),
        "zero_volume": sum(1 for r in rows if r[4] <= 0),
        "unsolved": 0,
        "note": "",
    }

    traded = [r for r in rows if r[4] > 0]
    if not traded:
        quality["note"] = "no traded options"
        return None, quality

    expiries = sorted({r[0] for r in traded
                       if (r[0] - day).days >= MIN_DAYS_TO_EXPIRY})
    if not expiries:
        quality["note"] = "no expiry far enough out"
        return None, quality

    near_chain = [r for r in traded if r[0] == expiries[0]]
    near = solve_atm_pair(near_chain, spot, day, expiries[0])
    if near is None:
        quality["note"] = "no usable ATM pair on the near expiry"
        return None, quality
    quality["unsolved"] = near["unsolved"]

    # The following expiry. The gap between the two is the event premium:
    # extra volatility priced into the expiry that covers an event, but not
    # into the weeks beyond it.
    nxt = None
    if len(expiries) > 1:
        next_chain = [r for r in traded if r[0] == expiries[1]]
        nxt = solve_atm_pair(next_chain, spot, day, expiries[1])

    # Skew: a put 5% below spot against a call 5% above, same near expiry.
    put_iv_below, put_strike = solve_near_strike(
        near_chain, spot * (1 - SKEW_OFFSET), PUT, spot, near["years"])
    call_iv_above, call_strike = solve_near_strike(
        near_chain, spot * (1 + SKEW_OFFSET), CALL, spot, near["years"])

    record = {
        "date": day,
        "spot": spot,
        "strike": near["strike"],
        "expiry": near["expiry"],
        "days_to_expiry": near["days_to_expiry"],
        "call_price": near["call_price"],
        "put_price": near["put_price"],
        "call_iv": near["call_iv"],
        "put_iv": near["put_iv"],
        "atm_iv": near["atm_iv"],
        "next_expiry": nxt["expiry"] if nxt else None,
        "next_iv": nxt["atm_iv"] if nxt else np.nan,
        "event_premium": (near["atm_iv"] - nxt["atm_iv"]) if nxt else np.nan,
        "skew_put_strike": put_strike,
        "skew_call_strike": call_strike,
        "skew": skew(put_iv_below * 100 if not np.isnan(put_iv_below) else np.nan,
                     call_iv_above * 100 if not np.isnan(call_iv_above) else np.nan),
        "expected_move_pct": expected_move_pct(
            near["call_price"], near["put_price"], spot),
    }
    return record, quality


def build_daily(spot_series, symbol=SYMBOL, only_days=None):
    """Everything above, for every trading day on disk.

    `only_days` restricts the work to a set of dates, which is what makes a
    nightly update take seconds instead of re-solving four years of options.

    The thing to notice is what it does NOT change. The filtering happens
    here, on the list of files; every day that is processed goes through the
    same read_day and atm_for_day as a full rebuild. So the fast path and the
    slow path cannot disagree about *how* a day is solved, only about which
    days they bother with -- and the weekly rebuild checks even that.
    """
    paths = sorted(RAW.glob("*.csv.zip"))
    if only_days is not None:
        wanted = set(only_days)
        paths = [p for p in paths
                 if dt.datetime.strptime(p.stem[:8], "%Y%m%d").date() in wanted]
    if not paths and only_days is None:
        sys.exit("no bhavcopies in data/raw -- run download.py first")

    records, quality_rows = [], []
    for i, path in enumerate(paths, 1):
        day = dt.datetime.strptime(path.stem[:8], "%Y%m%d").date()

        if day in THIN_SESSIONS:
            quality_rows.append({"date": day, "options": 0, "zero_volume": 0,
                                 "unsolved": 0,
                                 "note": "excluded: thin weekend session"})
            continue

        rows, underlying = read_day(path, symbol)
        if not rows:
            quality_rows.append({"date": day, "options": 0, "zero_volume": 0,
                                 "unsolved": 0, "note": f"no {symbol} options"})
            continue

        spot, source = None, ""
        if PREFER_UNDERLYING_FROM_BHAVCOPY and underlying:
            spot, source = underlying, "bhavcopy"
        else:
            ts = pd.Timestamp(day)
            if ts in spot_series.index:
                spot, source = float(spot_series.loc[ts]), "yahoo"

        if not spot or spot <= 0:
            quality_rows.append({"date": day, "options": len(rows),
                                 "zero_volume": 0, "unsolved": 0,
                                 "note": "no spot price available"})
            continue

        record, quality = atm_for_day(day, rows, spot)
        quality["spot_source"] = source
        quality_rows.append(quality)
        if record:
            record["spot_source"] = source
            records.append(record)

        if i % 100 == 0 or i == len(paths):
            print(f"  {i:>4}/{len(paths)} days  ({day})", flush=True)

    # A --fast run with nothing new to solve produces no rows at all, and an
    # empty DataFrame has no "date" column to sort by -- sorting it raises
    # KeyError. That is a normal state, not an error: it happens every time
    # this runs twice in a day, or on a holiday. The caller concatenates these
    # onto the saved series, which does have the columns, so an empty frame
    # with no columns at all concatenates cleanly.
    daily = pd.DataFrame(records)
    if not daily.empty:
        daily = daily.sort_values("date").reset_index(drop=True)

    quality_df = pd.DataFrame(quality_rows)
    if not quality_df.empty:
        quality_df = quality_df.sort_values("date").reset_index(drop=True)

    return daily, quality_df


# ---------------------------------------------------------------------------
# The event study
# ---------------------------------------------------------------------------

def realised_vol(spot_series, event_day, days=WINDOW):
    """Annualised volatility of Nifty over the `days` trading days after.

    Returns come from Yahoo's continuous close series rather than the
    bhavcopy underlying: a return series has to be measured the same way at
    both ends, and mixing two sources would invent jumps that never happened.
    """
    after = spot_series[spot_series.index > pd.Timestamp(event_day)]
    closes = pd.concat([
        spot_series[spot_series.index <= pd.Timestamp(event_day)].tail(1),
        after.head(days),
    ])
    if len(closes) < 6:                      # too short to mean anything
        return np.nan
    log_returns = np.log(closes / closes.shift(1)).dropna()
    return float(log_returns.std(ddof=1) * np.sqrt(252) * 100)


def event_study(daily, events, spot_series):
    """One row per event, the paths per event type, and a per-type summary."""
    calendar = list(daily["date"])
    index_of = {d: i for i, d in enumerate(calendar)}
    by_date = daily.set_index("date")

    results, curve_points = [], []
    event_trading_days = []

    for _, ev in events.iterrows():
        target = ev["date"].date()

        # Event day = first trading day on or after the event date. All 29
        # current events are trading days, but Budget and RBI news lands
        # during market hours, so a future event on a holiday must roll
        # forward to the first close that could react to it.
        on_or_after = [d for d in calendar if d >= target]
        if not on_or_after:
            print(f"  skipping {target} ({ev['event']}): beyond the data")
            continue
        event_day = on_or_after[0]
        i = index_of[event_day]
        if i == 0:
            print(f"  skipping {target}: no prior trading day")
            continue

        event_trading_days.append(event_day)
        before_day = calendar[i - 1]
        before, after = by_date.loc[before_day], by_date.loc[event_day]

        iv_before, iv_after = before["atm_iv"], after["atm_iv"]
        change = ((iv_after / iv_before - 1) * 100
                  if iv_before and not np.isnan(iv_before) else np.nan)
        rv = realised_vol(spot_series, event_day)

        # What the straddle implied the day before, against what Nifty did.
        # The straddle prices the move to expiry, so it must be scaled to one
        # day before being set against a one-day move -- see
        # analysis.expected_move_one_day for why this cuts both ways.
        expected = before["expected_move_pct"]
        expected_1d = expected_move_one_day(expected, before["days_to_expiry"])
        actual = abs(after["spot"] / before["spot"] - 1) * 100

        results.append({
            "event": ev["event"],
            "type": ev["type"],
            "event_date": target,
            "trading_day": event_day,
            "rolled": event_day != target,
            "iv_before": iv_before,
            "iv_after": iv_after,
            "change_pct": change,
            "realised_vol": rv,
            "overpriced": (bool(iv_before > rv)
                           if not (np.isnan(iv_before) or np.isnan(rv)) else None),
            "days_to_expiry_before": before["days_to_expiry"],
            "expected_move_to_expiry": expected,
            "expected_move_1d": expected_1d,
            "actual_move": actual,
            "move_overpriced": (bool(expected_1d > actual)
                                if not (np.isnan(expected_1d) or np.isnan(actual))
                                else None),
            "event_premium_before": before["event_premium"],
            "event_premium_after": after["event_premium"],
            "skew_before": before["skew"],
            "skew_after": after["skew"],
        })

        for offset in range(-WINDOW, WINDOW + 1):
            j = i + offset
            if 0 <= j < len(calendar):
                row = by_date.loc[calendar[j]]
                curve_points.append({
                    "type": ev["type"], "offset": offset,
                    "iv": row["atm_iv"], "next_iv": row["next_iv"],
                    "skew": row["skew"], "premium": row["event_premium"],
                })

    results_df = pd.DataFrame(results)

    curves = (pd.DataFrame(curve_points)
              .groupby(["type", "offset"])
              .agg(iv=("iv", "mean"), next_iv=("next_iv", "mean"),
                   skew=("skew", "mean"), premium=("premium", "mean"),
                   n=("iv", "count"))
              .reset_index())

    return results_df, curves, event_trading_days


def summarise(results_df):
    """Per-type averages, each with a bootstrap confidence range."""
    rows = []
    for event_type, group in results_df.groupby("type"):
        change_low, change_high = bootstrap_ci(group["change_pct"].to_numpy())
        iv_low, iv_high = bootstrap_ci(group["iv_before"].to_numpy())
        rows.append({
            "type": event_type,
            "n": len(group),
            "avg_iv_before": group["iv_before"].mean(),
            "iv_before_ci_low": iv_low,
            "iv_before_ci_high": iv_high,
            "avg_iv_after": group["iv_after"].mean(),
            "avg_change_pct": group["change_pct"].mean(),
            "change_ci_low": change_low,
            "change_ci_high": change_high,
            "avg_realised_vol": group["realised_vol"].mean(),
            "overpriced": int(group["overpriced"].sum()),
            "move_overpriced": int(group["move_overpriced"].sum()),
            "avg_expected_move_to_expiry": group["expected_move_to_expiry"].mean(),
            "avg_expected_move_1d": group["expected_move_1d"].mean(),
            "avg_actual_move": group["actual_move"].mean(),
            "avg_skew_before": group["skew_before"].mean(),
            "avg_premium_before": group["event_premium_before"].mean(),
        })
    return pd.DataFrame(rows)


def placebo(daily, results_df, event_trading_days):
    """Is each event type's average IV change unusual, or just a Tuesday?

    The pool is every one-day IV change on a day at least NORMAL_DAY_GAP
    trading days from any event, so no ordinary day overlaps the run-up or
    decay of an event we are measuring.
    """
    calendar = list(daily["date"])
    iv = daily["atm_iv"].to_numpy()

    changes = one_day_changes(iv)                    # one shorter than calendar
    is_normal = normal_day_mask(calendar, event_trading_days, gap=NORMAL_DAY_GAP)

    # changes[k] is the move landing on calendar[k + 1], so the mask for a
    # change is the mask of the day it lands on.
    normal_changes = changes[is_normal[1:]]
    normal_changes = normal_changes[~np.isnan(normal_changes)]

    rows = []
    for event_type, group in results_df.groupby("type"):
        event_changes = group["change_pct"].to_numpy()
        rows.append({
            "type": event_type,
            "n_events": len(group),
            "avg_change_pct": float(np.nanmean(event_changes)),
            "normal_days": len(normal_changes),
            "normal_mean_change_pct": float(normal_changes.mean()),
            "percentile_vs_normal": placebo_percentile(event_changes,
                                                       normal_changes),
        })
    return pd.DataFrame(rows), normal_changes


# ---------------------------------------------------------------------------

# Where each index's spot comes from for the pre-2024 files, which carry no
# UndrlygPric, and where its results are written.
#
# NIFTY writes to data/ as it always has, so every existing path keeps
# working. A second index gets its own folder rather than a suffix on six
# filenames, which keeps main.py's lookup a directory choice instead of a
# string-building exercise.
INDEXES = {
    "NIFTY":     {"spot": "spot.csv",      "out": ""},
    "BANKNIFTY": {"spot": "bank_spot.csv", "out": "banknifty"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default=SYMBOL, choices=sorted(INDEXES),
                    help=f"index to build (default {SYMBOL})")
    ap.add_argument("--fast", action="store_true",
                    help="only solve days missing from the existing "
                         "daily_iv.csv, then recompute everything downstream "
                         "from the full series")
    args = ap.parse_args()

    symbol = args.symbol
    config = INDEXES[symbol]
    out_dir = DATA / config["out"] if config["out"] else DATA
    out_dir.mkdir(parents=True, exist_ok=True)

    spot_path = DATA / config["spot"]
    if not spot_path.exists():
        sys.exit(f"{spot_path} is missing -- run download.py first")

    spot_series = (pd.read_csv(spot_path, parse_dates=["date"])
                   .set_index("date")["close"].sort_index())
    vix_series = (pd.read_csv(DATA / "vix.csv", parse_dates=["date"])
                  .set_index("date")["close"].sort_index())
    events = (pd.read_csv(DATA / "events.csv", parse_dates=["date"])
              .sort_values("date"))

    # The fast path solves only the days that are not already in
    # daily_iv.csv, then hands the complete series to the same event study,
    # placebo and summary code a full rebuild uses. Nothing downstream is
    # incremental -- those steps are pandas over a thousand rows and take a
    # moment, so there is no reason to approximate them.
    #
    # That is what keeps the two paths honest: the only difference is which
    # days get their options re-solved, and a solved day is a pure function
    # of its bhavcopy and its spot price. The weekly rebuild proves it.
    existing = None
    only_days = None
    if args.fast:
        prior = out_dir / "daily_iv.csv"
        qprior = out_dir / "quality.csv"
        if not prior.exists():
            sys.exit(f"{prior} is missing -- --fast updates an existing "
                     f"build. Run without --fast first.")
        existing = pd.read_csv(prior, parse_dates=["date", "expiry",
                                                   "next_expiry"])
        have = set(existing["date"].dt.date)
        # Reading a CSV gives Timestamps; a freshly solved day gives
        # datetime.date. Concatenating the two makes a column that cannot be
        # sorted, and writes dates in two different formats. Normalise the
        # saved rows to match what build_daily produces.
        for col in ("date", "expiry", "next_expiry"):
            if col in existing:
                existing[col] = existing[col].dt.date
        if qprior.exists():
            have |= set(pd.read_csv(qprior, parse_dates=["date"])["date"].dt.date)
        on_disk = {dt.datetime.strptime(p.stem[:8], "%Y%m%d").date()
                   for p in RAW.glob("*.csv.zip")}
        only_days = sorted(on_disk - have)
        print(f"{symbol}: fast update, {len(only_days)} new day(s) to solve")
        if not only_days:
            print("  nothing new -- recomputing the study from the existing "
                  "series anyway, in case events.csv changed")
    else:
        print(f"{symbol}: daily ATM IV from "
              f"{len(list(RAW.glob('*.csv.zip')))} bhavcopies")

    daily, quality = build_daily(spot_series, symbol, only_days)

    if existing is not None:
        qprior = out_dir / "quality.csv"
        if qprior.exists():
            quality = pd.concat(
                [pd.read_csv(qprior, parse_dates=["date"]), quality],
                ignore_index=True)
            quality["date"] = pd.to_datetime(quality["date"]).dt.date
            quality = (quality.drop_duplicates("date", keep="last")
                       .sort_values("date").reset_index(drop=True))
        # The new rows arrive without the vix column the old ones already
        # have; it is filled for the whole frame below, so drop it here to
        # keep the concat clean.
        old = existing.drop(columns=[c for c in ("vix",) if c in existing])
        daily = (pd.concat([old, daily], ignore_index=True)
                 .drop_duplicates("date", keep="last")
                 .sort_values("date").reset_index(drop=True))

    daily["vix"] = [
        float(vix_series.loc[pd.Timestamp(d)])
        if pd.Timestamp(d) in vix_series.index else np.nan
        for d in daily["date"]
    ]

    # Write the daily series now, then read it back and use *that* for
    # everything downstream.
    #
    # It looks redundant and it is deliberate. daily_iv.csv is written to four
    # decimals, so a full rebuild holding full precision in memory computes
    # the event study from slightly different numbers than a --fast run, which
    # reads the historical rows back from the file. The gap is about 7e-04 on
    # a percentage -- meaningless in itself, but it means the two paths can
    # never be compared by equality, only by an argued tolerance.
    #
    # Round-tripping through the file removes the question. Both paths now
    # compute from the exact numbers the website serves, so the weekly
    # rebuild can check they match byte for byte instead of "closely enough".
    daily_path = out_dir / "daily_iv.csv"
    daily.to_csv(daily_path, index=False, float_format="%.4f")
    daily = pd.read_csv(daily_path, parse_dates=["date", "expiry",
                                                 "next_expiry"])
    for col in ("date", "expiry", "next_expiry"):
        daily[col] = daily[col].dt.date

    print(f"\nEvent study over {len(events)} events")
    results, curves, event_days = event_study(daily, events, spot_series)
    summary = summarise(results)
    placebo_df, normal_changes = placebo(daily, results, event_days)

    for name, frame in (("daily_iv", daily), ("quality", quality),
                        ("results", results), ("curves", curves),
                        ("summary", summary), ("placebo", placebo_df)):
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.4f")
        print(f"  {path.name:<14} {len(frame):>5} rows")

    solved = daily["atm_iv"].notna().sum()
    corr = daily[["atm_iv", "vix"]].corr().iloc[0, 1]
    print(f"\n  {solved} of {len(quality)} days produced an ATM IV")
    print(f"  mean ATM IV {daily['atm_iv'].mean():.2f}%  "
          f"vs India VIX {daily['vix'].mean():.2f}  (corr {corr:.3f})")
    print(f"  next-expiry IV solved on {daily['next_iv'].notna().sum()} days, "
          f"skew on {daily['skew'].notna().sum()}")
    print(f"  placebo pool: {len(normal_changes)} ordinary days")


if __name__ == "__main__":
    main()
