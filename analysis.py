"""The statistics behind the event study.

Pure functions: arrays in, numbers out. No files, no dates, no I/O. That is
deliberate -- it means every one can be tested in milliseconds against inputs
whose answer you already know, instead of being verified by eye at the end of a
ten-minute run over 971 zip files.

build.py prepares the inputs and writes the outputs; this module only computes.

tests/test_analysis.py checks every function against cases whose answers are
known in advance, either from arithmetic you can do on paper or from a
distribution whose shape is fixed.
"""

import numpy as np

# Resamples for the bootstrap and the placebo test. 10,000 is the usual
# choice: enough that the answer is stable to about a tenth of a percentile,
# cheap enough to run on every build.
RESAMPLES = 10_000

# Fixed so two runs of build.py give identical numbers. A result that moves
# when you re-run it is a result nobody can check.
SEED = 20260101

# A day counts as "normal" only if it is at least this many trading days from
# every event. Ten matches the event window, so no normal day overlaps the
# run-up or decay of any event we are measuring.
NORMAL_DAY_GAP = 10


# ===========================================================================
# Confidence ranges
# ===========================================================================

def bootstrap_ci(values, level=0.95, resamples=RESAMPLES, seed=SEED):
    """A confidence interval for the mean of `values`, by resampling.

    Why this is needed: you have 5 Budgets. The average IV drop across them is
    a single number, and it says nothing about how much that number would move
    if history had dealt a slightly different five. The bootstrap answers that
    by building thousands of alternative histories out of the data you have.

    The method: draw `resamples` alternative samples from `values`, each the
    same size as the original and each drawn WITH REPLACEMENT. Average every
    one. The interval is the middle `level` of those averages.

    With replacement is the whole trick. Drawing 5 from 5 without replacement
    would return the same 5 events every time and every "resample" would have
    an identical mean. Allowing repeats is what simulates a different draw
    from the same underlying process.

    Expect a wide interval for Budget (n=5) and a narrow one for RBI (n=23).
    Saying so plainly in the README is the honest move, and it is exactly what
    this function exists to let you say.

    Returns
    -------
    (low, high) : tuple of float
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True)
    means = draws.mean(axis=1)

    low = np.percentile(means, 100 * (1 - level) / 2)
    high = np.percentile(means, 100 * (1 + level) / 2)
    return float(low), float(high)


# ===========================================================================
# The placebo test -- the most important function in the project
# ===========================================================================

def placebo_percentile(event_changes, normal_changes,
                       resamples=RESAMPLES, seed=SEED):
    """Where an event type's average IV change ranks among ordinary days.

    This is what turns "IV fell 5.6% on RBI days" into a claim worth making.
    On its own that number is meaningless: IV moves every day. The question is
    whether a drop that size is unusual, or just Tuesday.

    The method: take the real average change across the events. Then build a
    reference distribution by drawing groups of n ordinary days -- n matching
    the number of events -- and averaging each group. The answer is the share
    of those random group averages that fall below the real one.

    Read the answer like this: 3.0 means only 3% of random same-sized groups
    of ordinary days had an average drop as steep as the real events -- so the
    effect is real. 50.0 means the events look exactly like ordinary days, and
    there is no finding. Above 95 means the events rose unusually, which is
    what the June 2024 election should show.

    THE TRAP, and why this function is specified the way it is:

      The two project documents disagree. One says compare each event-day
      change against the pool of single normal days; the other says compare
      the event type's *average* against that same pool of single days. The
      second is wrong.

      The yardstick has to match the variability of whatever is being
      measured. What we measure is an average of n events, and an average of
      n values is sqrt(n) times LESS variable than a single value. Judging it
      against the spread of single days therefore uses a reference that is far
      too wide, and the observed average sits comfortably inside it looking
      unremarkable.

      So the error is conservative: it hides real effects rather than
      inventing them. With 23 RBI meetings and a 10% daily spread, an average
      drop that genuinely ranks at the 15th percentile reads as the 40th
      against single days -- and you would conclude there was nothing there.

      The fix is to compare like with like: build the reference distribution
      out of means of n ordinary days, where n matches the number of events.
      That is what the `size=(resamples, n)` draw below does, and it is the
      only reason this function needs resampling at all.

    Returns
    -------
    float : percentile in 0..100, or nan
    """
    event_changes = np.asarray(event_changes, dtype=float)
    event_changes = event_changes[~np.isnan(event_changes)]
    normal_changes = np.asarray(normal_changes, dtype=float)
    normal_changes = normal_changes[~np.isnan(normal_changes)]

    if len(event_changes) == 0 or len(normal_changes) < 30:
        return np.nan

    n = len(event_changes)
    observed = float(event_changes.mean())

    # n at a time, not one at a time: this is what makes the comparison
    # means-against-means rather than a mean against single days.
    rng = np.random.default_rng(seed)
    draws = rng.choice(normal_changes, size=(resamples, n), replace=True)
    means = draws.mean(axis=1)

    return float((means < observed).mean() * 100)


# ===========================================================================
# Expected move
# ===========================================================================

def expected_move_pct(call_price, put_price, spot):
    """How far the market expected Nifty to move, from the straddle price.

    Buying the at-the-money call and the at-the-money put together -- a
    straddle -- pays off if Nifty moves far enough in either direction. So
    what traders paid for that pair is, near enough, the move they were
    pricing in.

    This is a second, simpler check on the same question realised volatility
    answers. Compare it with how far Nifty actually moved by the event-day
    close: if the actual move is usually smaller, options were overpriced.
    Two independent routes to the same conclusion is much harder to argue
    with than one.

    It is an approximation, and knowing why matters. A straddle's exact
    expected absolute move is about 0.8 times its price under Black-Scholes
    (sqrt(2/pi), to be precise). The raw ratio is the convention on trading
    desks because it is quick and the bias is consistent, but do not claim
    more precision than it has.

    IMPORTANT: this is the move expected over the option's whole remaining
    life, not over one day. Comparing it with a single day's move is a
    horizon error -- see expected_move_one_day.

    Returns
    -------
    float : expected move as a percentage of spot, or nan
    """
    if spot <= 0 or np.isnan(call_price) or np.isnan(put_price):
        return np.nan
    return float((call_price + put_price) / spot * 100)


def expected_move_one_day(move_pct, days_to_expiry):
    """The share of an expected move that belongs to a single day.

    Volatility accumulates with the square root of time, so a move expected
    over T days implies roughly 1/sqrt(T) of it in any one of them.

    This correction matters a great deal here. The straddle the day before an
    RBI meeting has about 9 days left to run, so its raw expected move is
    about three times the one-day figure. Comparing the raw number against
    the event-day move -- which is what both project documents describe --
    makes options look overpriced on 22 of 23 RBI meetings. Corrected, it is
    11 of 23: no signal at all.

    But this correction has its own flaw, in the opposite direction. Dividing
    by sqrt(T) assumes volatility is spread evenly across the days remaining,
    and around an event it plainly is not -- the whole point is that traders
    expect the move to land on one particular day. So the true one-day implied
    move is somewhere above this figure.

    The honest conclusion is that the raw comparison overstates overpricing,
    this one understates it, and neither settles the question. Separating the
    event day's share of volatility from the rest needs a term-structure model
    the project does not have, so the realised-volatility comparison in the
    event study -- where both sides are annualised over the same horizon --
    is the measure to lead with.

    Returns
    -------
    float : one-day expected move as a percentage of spot, or nan
    """
    if np.isnan(move_pct) or days_to_expiry is None:
        return np.nan
    if days_to_expiry <= 0:
        return np.nan
    return float(move_pct / np.sqrt(days_to_expiry))


# ===========================================================================
# Given -- no need to write these
# ===========================================================================

def normal_day_mask(dates, event_dates, gap=NORMAL_DAY_GAP):
    """True for each date that is at least `gap` trading days from any event.

    `dates` must be the ordered trading-day calendar, so "gap" counts trading
    days rather than calendar days -- a weekend must not make a Monday look
    far from a Friday event.
    """
    dates = list(dates)
    index_of = {d: i for i, d in enumerate(dates)}
    blocked = set()
    for ev in event_dates:
        on_or_after = [d for d in dates if d >= ev]
        if not on_or_after:
            continue
        i = index_of[on_or_after[0]]
        blocked.update(range(max(0, i - gap), min(len(dates), i + gap + 1)))
    return np.array([i not in blocked for i in range(len(dates))])


def one_day_changes(series):
    """Percentage change from each value to the next, as an array.

    The first entry has nothing before it, so the result is one shorter than
    the input.
    """
    values = np.asarray(series, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (values[1:] / values[:-1] - 1.0) * 100.0


def skew(put_iv_below, call_iv_above):
    """Put IV minus call IV, both away from the money, in IV points.

    Positive means crash protection costs more than upside -- the normal state
    of an equity index. Watching it over an event window shows whether that
    premium widens before events.
    """
    if np.isnan(put_iv_below) or np.isnan(call_iv_above):
        return np.nan
    return float(put_iv_below - call_iv_above)
