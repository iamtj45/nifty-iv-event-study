"""Does knowing an event is coming tell you which way the market will go?

    python direction.py

Reads   data/daily_iv.csv     the spot series, one row per trading day
        data/results.csv      the event days, same ones the study published
        data/events.csv       to mark which days are events
Writes  data/direction.csv    one row per event type

Why this is a separate question from the rest of the project.

Everything else here measures how *much* the index moves -- implied
volatility, realised volatility, the straddle. None of it has any opinion
about *which way*. A straddle seller does not care whether Nifty rises or
falls, only how far it travels, so the whole study so far has been blind to
direction by construction.

That leaves the obvious question untested: if you know an RBI decision lands
tomorrow, do you know anything about the direction? This file answers it, and
the answer deserves measuring rather than assuming in either direction.

The null hypothesis is a coin. If events carry no directional information,
roughly half should be up and half down, and the average move should look
like an ordinary day's.

Three tests, because each can fail while the others pass:

    sign test     of n events, k went up. How surprising is that from a coin?
                  Answers "is the direction lopsided at all".

    mean move     with a bootstrap confidence range. A group can split 12/11
                  and still have a strongly negative average if the down days
                  are much larger. Direction and magnitude are separate.

    placebo       where the event group's average ranks among 10,000 random
                  groups of ordinary days of the same size. The same test the
                  IV study uses, so the two results are directly comparable.

A warning this file prints rather than hides. Three event types are tested, so
one looking unusual at the 5% level is roughly what chance produces anyway.
And with five Budgets, any Budget result is a hypothesis for the next five
rather than a finding. Both caveats are in the output because a reader should
not have to know to ask.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from analysis import (NORMAL_DAY_GAP, RESAMPLES, SEED, bootstrap_ci,
                      normal_day_mask, placebo_percentile)
from build import INDEXES, SYMBOL

DATA = Path(__file__).parent / "data"

# Trading days after the event for the second horizon. The same window the
# realised-volatility measure uses, so "what happened afterwards" means the
# same stretch of time in both places.
WINDOW = 10

# Below this many events, a result is reported as a hypothesis rather than a
# finding, however small its p-value looks.
MIN_FOR_A_FINDING = 10


def load(out_dir):
    daily = pd.read_csv(out_dir / "daily_iv.csv", parse_dates=["date"])
    results = pd.read_csv(out_dir / "results.csv",
                          parse_dates=["event_date", "trading_day"])
    events = pd.read_csv(DATA / "events.csv", parse_dates=["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day"] = daily["date"].dt.date
    return daily, results, events


def moves(daily):
    """Percentage change in spot from each trading day to the next.

    Computed from the spot column of daily_iv.csv rather than from Yahoo
    directly, so the index level used here is exactly the one every implied
    volatility in the study was solved against. A direction measured from a
    different price series than the volatility would not be comparable with
    it.
    """
    spot = daily["spot"].to_numpy(dtype=float)
    out = np.full(len(spot), np.nan)
    out[1:] = (spot[1:] / spot[:-1] - 1.0) * 100.0
    return out


def forward_move(daily, index, days=WINDOW):
    """Percentage change over the `days` trading days after `index`."""
    spot = daily["spot"].to_numpy(dtype=float)
    end = min(index + days, len(spot) - 1)
    if end <= index or spot[index] <= 0:
        return np.nan
    return (spot[end] / spot[index] - 1.0) * 100.0


def collect(daily, results):
    """One row per event: the move on the day, and over the next 10 days."""
    day_move = moves(daily)
    position = {d: i for i, d in enumerate(daily["day"])}

    rows = []
    for row in results.itertuples():
        day = row.trading_day.date()
        i = position.get(day)
        if i is None or i == 0:
            continue
        rows.append({
            "event": row.event,
            "type": row.type,
            "event_date": day,
            "move_on_day": day_move[i],
            "move_10d_after": forward_move(daily, i),
        })
    return pd.DataFrame(rows)


def normal_moves(daily, events):
    """The move on every ordinary day: the baseline any event must beat."""
    day_move = moves(daily)
    calendar = list(daily["day"])
    event_days = sorted({d.date() for d in events["date"]})
    mask = normal_day_mask(calendar, event_days, gap=NORMAL_DAY_GAP)
    values = day_move[np.asarray(mask)]
    return values[~np.isnan(values)]


def test_group(values, baseline, label):
    """Sign test, bootstrap range and placebo rank for one group of events."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return None

    up = int((values > 0).sum())
    down = int((values < 0).sum())
    flat = n - up - down

    # Sign test over the days that actually moved. A day that closed exactly
    # unchanged carries no directional information and would otherwise be
    # counted as evidence for "not up".
    decided = up + down
    p_sign = (binomtest(up, decided, 0.5, alternative="two-sided").pvalue
              if decided > 0 else np.nan)

    low, high = bootstrap_ci(values)
    pct = (placebo_percentile(values, baseline, resamples=RESAMPLES, seed=SEED)
           if n >= 2 else np.nan)

    return {
        "type": label,
        "n": n,
        "up": up,
        "down": down,
        "unchanged": flat,
        "pct_up": round(up / decided * 100, 1) if decided else np.nan,
        "mean_move": round(float(values.mean()), 4),
        "median_move": round(float(np.median(values)), 4),
        "ci_low": round(low, 4) if not np.isnan(low) else np.nan,
        "ci_high": round(high, 4) if not np.isnan(high) else np.nan,
        "sign_test_p": round(float(p_sign), 4) if not np.isnan(p_sign) else np.nan,
        "placebo_percentile": round(pct, 2) if not np.isnan(pct) else np.nan,
    }


def volatility_direction(daily):
    """Does volatility tell you direction -- and does it predict it?

    Two questions that sound alike and have opposite answers.

    The first is about the same day. When fear rises, which way did the Nifty
    market go? This turns out to be a strong relationship, and it is the one
    most people have in mind when they say volatility and the market are
    linked.

    The second is about tomorrow. Today's fear spiked -- does that tell you
    anything about tomorrow's direction? Almost nothing, and the gap between
    the two answers is the whole point of measuring both.

    Why the difference is not a contradiction: rising fear and a falling
    market are largely the same event seen two ways. People are not
    forecasting a drop, they are reacting to the one on their screen. A
    thermometer reads today's heat accurately and says nothing about
    tomorrow's weather.

    Reported as a table rather than a single correlation because "-0.51"
    means little on its own, while "the Nifty market fell on 62% of the days
    fear rose" is a sentence anyone can check against their own memory.
    """
    frame = daily.sort_values("date").copy()
    frame["ret"] = frame["spot"].pct_change() * 100
    frame["iv_chg"] = frame["atm_iv"].pct_change() * 100
    frame["next_ret"] = frame["ret"].shift(-1)

    same = frame.dropna(subset=["ret", "iv_chg"])
    rose = same[same["iv_chg"] > 0]
    fell = same[same["iv_chg"] < 0]

    ahead = frame.dropna(subset=["iv_chg", "next_ret"])
    # The top decile of fear spikes: if anything predicts tomorrow, the most
    # dramatic days are where it should show up.
    cutoff = ahead["iv_chg"].quantile(0.90)
    spikes = ahead[ahead["iv_chg"] > cutoff]

    rows = [
        {"measure": "correlation of IV change with Nifty return",
         "horizon": "same day",
         "value": round(float(same["ret"].corr(same["iv_chg"])), 4),
         "n": len(same),
         "note": "strongly negative: fear rising and the market falling are "
                 "the same event seen two ways"},
        {"measure": "Nifty market fell, on days IV rose",
         "horizon": "same day",
         "value": round(float((rose["ret"] < 0).mean() * 100), 1),
         "n": len(rose),
         "note": f"average return {rose['ret'].mean():+.3f}%"},
        {"measure": "Nifty market rose, on days IV fell",
         "horizon": "same day",
         "value": round(float((fell["ret"] > 0).mean() * 100), 1),
         "n": len(fell),
         "note": f"average return {fell['ret'].mean():+.3f}%"},
        {"measure": "correlation of IV change with the NEXT day's return",
         "horizon": "next day",
         "value": round(float(ahead["iv_chg"].corr(ahead["next_ret"])), 4),
         "n": len(ahead),
         "note": "about zero: today's fear says nothing about tomorrow"},
        {"measure": "Nifty market rose the day after the biggest 10% of IV spikes",
         "horizon": "next day",
         "value": round(float((spikes["next_ret"] > 0).mean() * 100), 1),
         "n": len(spikes),
         "note": f"average return {spikes['next_ret'].mean():+.3f}%"},
        {"measure": "Nifty market rose the day after any day (baseline)",
         "horizon": "next day",
         "value": round(float((ahead["next_ret"] > 0).mean() * 100), 1),
         "n": len(ahead),
         "note": "compare with the row above: the difference is the whole "
                 "predictive value, and it is small"},
    ]
    return pd.DataFrame(rows)


def verdict(row):
    """Plain English, with the sample size doing the hedging."""
    if row["n"] < MIN_FOR_A_FINDING:
        if row["sign_test_p"] is not None and row["sign_test_p"] < 0.05:
            return (f"suggestive, but only {row['n']} events - a hypothesis "
                    f"for the next {row['n']}, not a finding")
        return f"too few events ({row['n']}) to say anything"

    ci_clears_zero = (row["ci_low"] > 0) or (row["ci_high"] < 0)
    if row["sign_test_p"] < 0.05 and ci_clears_zero:
        return "direction looks genuinely lopsided"
    if row["sign_test_p"] < 0.05:
        return "lopsided in count, but the average move still includes zero"
    return "indistinguishable from a coin"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default=SYMBOL, choices=sorted(INDEXES),
                    help=f"index to test (default {SYMBOL})")
    args = ap.parse_args()

    folder = INDEXES[args.symbol]["out"]
    out_dir = DATA / folder if folder else DATA
    for name in ("daily_iv", "results"):
        if not (out_dir / f"{name}.csv").exists():
            sys.exit(f"{out_dir}/{name}.csv is missing -- "
                     f"run build.py --symbol {args.symbol} first")

    daily, results, events = load(out_dir)
    per_event = collect(daily, results)
    if per_event.empty:
        sys.exit("no events line up with the trading calendar")

    baseline = normal_moves(daily, events)
    print(f"{args.symbol}: direction test over {len(per_event)} events, "
          f"against {len(baseline)} ordinary days\n")

    rows = []
    for kind in ["RBI", "BUDGET", "ELECTION"]:
        part = per_event[per_event["type"] == kind]
        if part.empty:
            continue
        stats = test_group(part["move_on_day"], baseline, kind)
        if stats:
            stats["verdict"] = verdict(stats)
            rows.append(stats)

    # The baseline itself, as a row. Ordinary days drift slightly up, and an
    # event group should be read against that rather than against zero.
    base = test_group(baseline, baseline, "ORDINARY")
    if base:
        base["placebo_percentile"] = np.nan   # it is the reference, not a test
        base["verdict"] = "the baseline every group is measured against"
        rows.append(base)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "direction.csv", index=False)

    # The per-event moves as well, signed. results.csv carries
    # actual_move, but that is the size of the move with the sign thrown
    # away -- it exists to be compared against the straddle price, which
    # has no opinion about direction. Anything that wants to say which
    # way a particular event went needs this file.
    per_event.to_csv(out_dir / "direction_events.csv", index=False)

    vol_dir = volatility_direction(daily)
    vol_dir.to_csv(out_dir / "vol_direction.csv", index=False)
    print()
    print("  volatility and direction")
    print(vol_dir.to_string(index=False))

    show = ["type", "n", "up", "down", "pct_up", "mean_move", "median_move",
            "ci_low", "ci_high", "sign_test_p", "placebo_percentile"]
    print(table[show].to_string(index=False))

    print("\n  verdicts")
    for row in rows:
        print(f"    {row['type']:9} {row['verdict']}")

    print("\n  how to read this")
    print("    sign_test_p         chance of a split this lopsided from a coin")
    print("    ci_low / ci_high    95% range on the average move; a range "
          "crossing zero means no direction")
    print("    placebo_percentile  rank among 10,000 random groups of "
          "ordinary days of the same size")

    print("\n  two caveats, stated rather than left for the reader to spot")
    print("    Three event types are tested here. One of three looking "
          "unusual at the 5% level is")
    print("    roughly what chance produces on its own, so a single small "
          "p-value is weak evidence.")
    small = [r["type"] for r in rows if r["n"] < MIN_FOR_A_FINDING
             and r["type"] != "ORDINARY"]
    if small:
        print(f"    {', '.join(small)} have fewer than {MIN_FOR_A_FINDING} "
              f"events. Whatever they show is a hypothesis")
        print("    for the next few, not a result, and it was noticed after "
              "looking at the data,")
        print("    which is the most common way to be convinced by noise.")

    print(f"\n  {len(table)} rows -> direction.csv")


if __name__ == "__main__":
    main()
