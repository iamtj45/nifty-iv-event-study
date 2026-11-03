"""Would selling the straddle before events actually have made money?

    python backtest.py
    python backtest.py --cost 4.0        assume a worse round-trip cost
    python backtest.py --no-normal       events only, skip the 448-day control

Reads   data/daily_iv.csv     the entry leg: strike, expiry, call and put close
        data/results.csv      the event days, same ones the study published
        data/events.csv       to mark which days are events
        data/raw/*.csv.zip    the exit leg, read one day at a time
Writes  data/backtest.csv           one row per trade
        data/backtest_summary.csv   one row per group

Why this file exists.

Phase 2 found that options were overpriced before 23 of 29 events, and phase 3
showed that is not chance. The question that follows immediately is whether
you could have taken the other side and been paid for it. Until this file
runs, the honest answer is "we have not measured that".

The rule, fixed in advance and never varied:

    sell the at-the-money straddle at the close the day before the event
    buy it back at the close on the event day

A straddle is one call and one put at the same strike. Selling it means you
collect both premiums, and you keep them if the index does not move far.
That is the trade that profits from implied volatility being too high, so it
is the trade that tests the finding directly.

Fixing the rule in advance matters. If you let yourself choose the strike, the
expiry or the exit after seeing the data, you will find a profit in almost any
dataset -- that is not a backtest, it is a search. One rule, applied
identically to every event and to every ordinary day, is the only version
that means anything.

The control group is the 448 ordinary days used by the placebo test -- days at
least ten trading days from any event. The same rule is run on each of them.
If selling before events pays no better than selling on a random Tuesday,
there is no event edge, whatever the average profit happens to be.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import NORMAL_DAY_GAP, bootstrap_ci, normal_day_mask
from build import INDEXES, RAW, SYMBOL, read_day
from iv import CALL, PUT

DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Decision 1: profit is measured in index points, not rupees.
#
# The Nifty option lot size changed twice over this period -- 25, then 50,
# then 75 contracts per lot. A rupee profit from 2023 and one from 2026 are
# therefore not comparable quantities, and averaging them would silently
# weight the later trades three times as heavily.
#
# One straddle, one unit of the index, points throughout. Multiply by whatever
# the lot size was on the day if you want rupees. Every trade is also reported
# as a percentage of the premium collected, which is the figure that answers
# "how much of what I sold did I keep?" and is comparable across the years.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Decision 2: the cost model, and why it is split in two.
#
# Costs are an assumption -- nobody publishes what you would actually have
# paid. Rather than hide one number in the code, there are two, because the
# two behave differently:
#
#   COST_POINTS   the bid-ask spread you cross and the flat brokerage. Does
#                 not scale with the premium: you pay it whether the straddle
#                 is worth 80 points or 400. This is the one most backtests
#                 quietly omit, and it is usually the largest.
#
#   COST_PCT      STT, exchange transaction charges, GST, stamp duty. These
#                 are a percentage of the premium traded, so they grow with
#                 the position.
#
# Both are deliberately set on the pessimistic side. A backtest that flatters
# itself is worth nothing, and the summary reports the break-even cost anyway
# -- the figure that says how wrong these assumptions would have to be to
# erase the result.
# ---------------------------------------------------------------------------
COST_POINTS = 2.0      # spread + brokerage, round trip, in index points
COST_PCT = 0.0015      # 0.15% of premium traded, both sides, taxes and charges


def load_inputs(out_dir):
    """The three result files, plus the trading-day calendar.

    events.csv is read from data/ whatever the index -- the dates of RBI
    meetings and Budgets do not depend on which options you are pricing.
    """
    daily = pd.read_csv(out_dir / "daily_iv.csv",
                        parse_dates=["date", "expiry"])
    results = pd.read_csv(out_dir / "results.csv",
                          parse_dates=["event_date", "trading_day"])
    events = pd.read_csv(DATA / "events.csv", parse_dates=["date"])

    daily["day"] = daily["date"].dt.date
    daily = daily.sort_values("day").reset_index(drop=True)
    return daily, results, events


# ===========================================================================
# WRITE THIS ONE  (1 of 3)
# ===========================================================================

def exit_quote(day, strike, expiry, symbol=SYMBOL):
    r"""The close of the call and put you sold, on the day you buy them back.

    This is the function the whole backtest rests on, and the one place it is
    easy to produce a flattering number by accident.

    You sold a specific contract: one strike, one expiry. To close the
    position you must buy back *that* contract. On the event day the index has
    usually moved, so the strike that is now at-the-money is a different
    strike -- and pricing your exit off that one would mean closing a position
    you never opened. It would also bias the result in a predictable
    direction, because the new at-the-money straddle is cheaper than the one
    that has drifted away from the money, making every trade look profitable.

    Matching the expiry matters for the same reason. The same strike exists on
    several expiry dates at once, and they are completely different
    instruments with completely different prices.

    The method:

      1. Build the path: RAW / f"{day:%Y%m%d}.csv.zip". If it does not
         exist, return None -- that day was not a trading day, or was never
         downloaded.
      2. rows, _ = read_day(path). Each row is a 5-tuple:
             (expiry, strike, kind, close, volume)
         read_day already keeps only NIFTY options, so there is no filtering
         by symbol to do here.
      3. Keep the rows where the expiry equals `expiry` AND the strike equals
         `strike` AND the volume is above zero. Requiring volume is the same
         standard build.py applies: the exchange publishes a close for
         contracts that never traded, and that number is not a price anyone
         could have dealt at.
      4. From those, find the call and the put -- row[2] is the kind, and
         iv.py gives you CALL and PUT for comparing against it.
      5. If either leg is missing, return None. Half a straddle cannot be
         closed, and guessing the other half would be inventing a price.
      6. Otherwise return (call_close, put_close) as floats -- that is row[3]
         from each.

    Beware the comparison in step 3. `expiry` from daily_iv.csv arrives as a
    pandas Timestamp, while read_day returns a datetime.date. Comparing those
    two directly is False even when they are the same day. Convert one side.

    Parameters
    ----------
    day : datetime.date       the day you close the position
    strike : float            the strike you sold
    expiry : date/Timestamp   the expiry you sold

    Returns
    -------
    (call_close, put_close) or None
    """
    path = RAW / f"{day:%Y%m%d}.csv.zip"
    if not path.exists():
        return None

    rows, _ = read_day(path, symbol)
    want = expiry.date() if hasattr(expiry, "date") else expiry

    legs = {}
    for row_expiry, row_strike, kind, close, volume in rows:
        if row_expiry == want and row_strike == strike and volume > 0:
            legs[kind] = close

    if CALL not in legs or PUT not in legs:
        return None
    return float(legs[CALL]), float(legs[PUT])


# ===========================================================================
# WRITE THIS ONE  (2 of 3)
# ===========================================================================

def straddle_trade(entry, exit_prices, cost_points=COST_POINTS,
                   cost_pct=COST_PCT):
    """One complete sell-and-buy-back trade, in index points.

    You collected two premiums yesterday and paid two back today. The profit
    is the difference, less costs.

    The sign is the thing to get right, and it is the opposite of the instinct
    built up from buying things. You are the seller. Money came in first. So
    you profit when the price you buy back at is *lower* than the price you
    sold at:

        gross = premium_received - premium_paid

    If implied volatility collapses after the event -- which phase 2 measured
    at about -5.6% for RBI days -- the straddle gets cheaper and this number
    is positive. If the index jumps instead, the straddle gets dearer and you
    lose. Getting this backwards would turn the entire result upside down and
    nothing in the code would complain, because a negative number is a
    perfectly valid loss.

    The method:

      1. premium_received = entry["call_price"] + entry["put_price"]
      2. premium_paid = exit_prices[0] + exit_prices[1]
      3. gross = premium_received - premium_paid
      4. Costs. Two parts, as described at the top of this file:
             cost = cost_points + cost_pct * (premium_received + premium_paid)
         The percentage applies to both sides because you are charged on the
         premium you trade, and you traded it twice.
      5. net = gross - cost
      6. Return a dict with, at least:
             premium_received, premium_paid, gross, cost, net,
             pct_of_premium  = net / premium_received * 100
         Guard that last one against a zero premium.

    Why pct_of_premium earns its place: a net of +30 points means something
    very different on a straddle sold for 400 than on one sold for 90. The
    percentage is what makes a 2023 trade and a 2026 trade comparable, and it
    is the figure to quote when someone asks how good the trade was.

    Returns
    -------
    dict
    """
    premium_received = float(entry["call_price"]) + float(entry["put_price"])
    premium_paid = float(exit_prices[0]) + float(exit_prices[1])

    gross = premium_received - premium_paid
    cost = cost_points + cost_pct * (premium_received + premium_paid)
    net = gross - cost

    return {
        "premium_received": premium_received,
        "premium_paid": premium_paid,
        "gross": gross,
        "cost": cost,
        "net": net,
        "pct_of_premium": (net / premium_received * 100
                           if premium_received > 0 else np.nan),
    }


# ---------------------------------------------------------------------------
# Running the rule over a set of days
# ---------------------------------------------------------------------------

def run_one(daily, exit_day, cost_points, cost_pct, symbol=SYMBOL):
    """Apply the rule to one exit day, or return None with a reason.

    The entry is the previous row of daily_iv.csv -- the previous *trading*
    day, not yesterday's calendar date, which matters across weekends and the
    57 holidays in this period.
    """
    rows = daily.index[daily["day"] == exit_day]
    if len(rows) == 0:
        return None, "no usable option data on the exit day"
    i = rows[0]
    if i == 0:
        return None, "no trading day before it in the data"

    entry = daily.loc[i - 1]
    if pd.isna(entry["call_price"]) or pd.isna(entry["put_price"]):
        return None, "entry day has no ATM pair"

    quote = exit_quote(exit_day, float(entry["strike"]), entry["expiry"],
                       symbol)
    if quote is None:
        return None, "the contract sold did not trade on the exit day"

    trade = straddle_trade(entry, quote, cost_points, cost_pct)
    exit_row = daily.loc[i]

    trade.update({
        "entry_day": entry["day"],
        "exit_day": exit_day,
        "strike": float(entry["strike"]),
        "expiry": entry["expiry"].date(),
        "days_to_expiry": int(entry["days_to_expiry"]),
        "entry_spot": float(entry["spot"]),
        "exit_spot": float(exit_row["spot"]),
        "index_move_pct": (float(exit_row["spot"]) / float(entry["spot"]) - 1) * 100,
        "entry_iv": float(entry["atm_iv"]),
        "exit_iv": float(exit_row["atm_iv"]),
    })
    return trade, ""


def run_trades(daily, results, events, cost_points, cost_pct,
               do_normal=True, symbol=SYMBOL):
    """The rule on every event day, and on every ordinary day for comparison."""
    trades, skipped = [], []

    for row in results.itertuples():
        exit_day = row.trading_day.date()
        trade, why = run_one(daily, exit_day, cost_points, cost_pct, symbol)
        if trade is None:
            skipped.append(f"    {row.event} ({exit_day}): {why}")
            continue
        trade["group"] = row.type
        trade["event"] = row.event
        trades.append(trade)

    if do_normal:
        calendar = list(daily["day"])
        event_days = sorted({d.date() for d in events["date"]})
        mask = normal_day_mask(calendar, event_days, gap=NORMAL_DAY_GAP)
        normal_days = [d for d, keep in zip(calendar, mask) if keep]
        print(f"  control group: {len(normal_days)} ordinary days "
              f"(at least {NORMAL_DAY_GAP} trading days from any event)")

        for n, day in enumerate(normal_days, 1):
            trade, _ = run_one(daily, day, cost_points, cost_pct, symbol)
            if trade is None:
                continue
            trade["group"] = "NORMAL"
            trade["event"] = ""
            trades.append(trade)
            if n % 100 == 0 or n == len(normal_days):
                print(f"    {n:>4}/{len(normal_days)} ordinary days", flush=True)

    return pd.DataFrame(trades), skipped


# ===========================================================================
# WRITE THIS ONE  (3 of 3)
# ===========================================================================

def summarise_trades(frame):
    r"""One row per group: is this a strategy, or a story about one good year?

    An average profit on its own is close to worthless, and this function
    exists to say the three things that give it meaning.

    **Win rate.** Short volatility is a trade that wins often and loses
    badly. A rule that wins 85% of the time and loses eight times the average
    win on the other 15% is a losing rule that looks excellent in the average
    until the day it does not. Average and win rate have to be read together.

    **The worst single loss.** The most important number in this file. Your
    plan names the day: 4 June 2024, the election result, when Nifty moved
    sharply -- precisely the day that ruins a straddle seller. If the worst
    loss is several times the average profit, the honest conclusion is that
    the edge is real but not harvestable with this rule, and saying so is
    worth more than a profitable-looking average.

    **The break-even cost.** The cost assumption is a guess, so report how
    much it would take to erase the result: the average gross profit, in
    points, is exactly the round-trip cost at which the strategy stops making
    money. If that is 1.2 points, the result is inside the noise of the
    bid-ask spread and there is nothing here. If it is 25 points, the finding
    survives costs being badly underestimated.

    The method -- group by the "group" column, and for each group build:

      n                  number of trades
      avg_net            mean of "net"
      median_net         median of "net", which a single disaster cannot drag
      avg_pct            mean of "pct_of_premium"
      win_rate           percentage of trades with net > 0
      worst              the minimum of "net"
      worst_day          the exit_day of that worst trade
      best               the maximum of "net"
      avg_gross          mean of "gross" -- the break-even round-trip cost
      ci_low, ci_high    bootstrap_ci(group["net"]) from analysis.py, a 95%
                         range on the average, since 5 Budgets cannot pin an
                         average down and the output should admit it

    Build the rows in a list and return a DataFrame. Put NORMAL last if you
    like -- it reads better as the baseline underneath the event groups.

    Returns
    -------
    DataFrame
    """
    rows = []
    groups = set(frame["group"])
    order = [g for g in ("RBI", "BUDGET", "ELECTION") if g in groups]
    order += [g for g in sorted(groups) if g not in order]

    for group in order:
        part = frame[frame["group"] == group]
        net = part["net"].astype(float)
        worst_row = part.loc[net.idxmin()]
        low, high = bootstrap_ci(net.to_numpy())

        rows.append({
            "group": group,
            "n": len(part),
            "avg_net": round(float(net.mean()), 2),
            "median_net": round(float(net.median()), 2),
            "avg_pct": round(float(part["pct_of_premium"].mean()), 2),
            "win_rate": round(float((net > 0).mean() * 100), 1),
            "worst": round(float(net.min()), 2),
            "worst_day": worst_row["exit_day"],
            "best": round(float(net.max()), 2),
            "avg_gross": round(float(part["gross"].mean()), 2),
            "ci_low": round(low, 2) if not np.isnan(low) else np.nan,
            "ci_high": round(high, 2) if not np.isnan(high) else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=float, default=COST_POINTS,
                        help=f"round-trip cost in index points "
                             f"(default {COST_POINTS})")
    parser.add_argument("--cost-pct", type=float, default=COST_PCT,
                        help=f"charges as a fraction of premium traded "
                             f"(default {COST_PCT})")
    parser.add_argument("--no-normal", action="store_true",
                        help="skip the ordinary-day control group")
    parser.add_argument("--symbol", default=SYMBOL, choices=sorted(INDEXES),
                        help=f"index to backtest (default {SYMBOL})")
    args = parser.parse_args()

    if not RAW.exists() or not any(RAW.glob("*.csv.zip")):
        sys.exit("no bhavcopies in data/raw -- run download.py first")

    symbol = args.symbol
    folder = INDEXES[symbol]["out"]
    out_dir = DATA / folder if folder else DATA
    if not (out_dir / "daily_iv.csv").exists():
        sys.exit(f"{out_dir / 'daily_iv.csv'} is missing -- "
                 f"run build.py --symbol {symbol} first")

    daily, results, events = load_inputs(out_dir)
    print(f"{symbol}: straddle backtest over {len(results)} events")
    print(f"  cost assumption: {args.cost} points + "
          f"{args.cost_pct * 100:.3f}% of premium traded\n")

    trades, skipped = run_trades(daily, results, events,
                                 args.cost, args.cost_pct,
                                 do_normal=not args.no_normal,
                                 symbol=symbol)
    if trades.empty:
        sys.exit("no trades could be built -- check the gaps are filled in")

    if skipped:
        print("\n  events skipped:")
        for line in skipped:
            print(line)

    summary = summarise_trades(trades)

    trades = trades.sort_values(["group", "exit_day"])
    trades.to_csv(out_dir / "backtest.csv", index=False)
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)

    print(f"\n  {len(trades)} trades -> backtest.csv")
    print(f"  {len(summary)} groups -> backtest_summary.csv\n")

    show = ["group", "n", "avg_net", "win_rate", "worst", "worst_day",
            "avg_gross"]
    show = [c for c in show if c in summary.columns]
    print(summary[show].to_string(index=False))


if __name__ == "__main__":
    main()
