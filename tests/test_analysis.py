"""Tests for analysis.py -- the three functions you write.

    pytest tests/test_analysis.py -v

Every case has an answer known in advance, either from arithmetic you can do
on paper or from a distribution whose shape is fixed. Until the stubs are
filled in these fail with NotImplementedError; that is the point.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import (bootstrap_ci, expected_move_pct, normal_day_mask,
                      one_day_changes, placebo_percentile, skew)


# ===========================================================================
# bootstrap_ci
# ===========================================================================

def test_ci_brackets_the_mean():
    """The interval must contain the sample mean. If it does not, the
    resampling is wrong -- bootstrap means centre on the sample mean."""
    values = [-14.7, -28.5, -25.1, -22.9, 10.6]        # the five Budgets
    low, high = bootstrap_ci(values)
    assert low < np.mean(values) < high


def test_ci_is_ordered_and_finite():
    low, high = bootstrap_ci([-5.0, -6.0, -4.0, -5.5, -4.5, -5.2])
    assert low < high
    assert math.isfinite(low) and math.isfinite(high)


def test_ci_narrows_as_the_sample_grows():
    """Twenty-three events must give a tighter interval than five.

    This is the whole reason the function exists: it is what lets the README
    say the Budget average is rough and the RBI average is not."""
    rng = np.random.default_rng(1)
    small = rng.normal(-5, 4, 5)
    large = rng.normal(-5, 4, 23)
    lo_s, hi_s = bootstrap_ci(small)
    lo_l, hi_l = bootstrap_ci(large)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_ci_on_identical_values_is_a_point():
    """With no spread in the data there is no uncertainty in the mean."""
    low, high = bootstrap_ci([7.0] * 10)
    assert low == pytest.approx(7.0)
    assert high == pytest.approx(7.0)


def test_ci_width_tracks_the_confidence_level():
    values = list(np.random.default_rng(2).normal(0, 5, 30))
    lo95, hi95 = bootstrap_ci(values, level=0.95)
    lo50, hi50 = bootstrap_ci(values, level=0.50)
    assert (hi50 - lo50) < (hi95 - lo95)


def test_ci_handles_nans_and_tiny_samples():
    low, high = bootstrap_ci([1.0, np.nan, 3.0])
    assert low < 2.0 < high                            # NaN dropped, n=2
    assert all(math.isnan(v) for v in bootstrap_ci([5.0]))
    assert all(math.isnan(v) for v in bootstrap_ci([]))
    assert all(math.isnan(v) for v in bootstrap_ci([np.nan, np.nan]))


def test_ci_is_reproducible():
    """Same seed, same answer. A number that moves between runs is a number
    nobody can check."""
    values = [-1.0, -9.0, 2.0, -4.0, -7.0]
    assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)


# ===========================================================================
# placebo_percentile  -- the important one
# ===========================================================================

def test_placebo_is_unbiased_and_spreads():
    """Events drawn from the same pool as ordinary days must look ordinary.

    One draw proves nothing: a random 20 can sit 1.3 standard errors from the
    pool mean, which honestly ranks at the 90th percentile. The property worth
    testing is over many draws -- the percentiles should centre on 50 and
    spread out roughly uniformly.

    The spread is what discriminates. A uniform distribution on 0..100 has a
    standard deviation near 28.9. If the reference were built from single days
    instead of means of n, it would be sqrt(n) times too wide, every observed
    mean would land near its middle, and these percentiles would cluster
    tightly around 50 with a standard deviation around 8.
    """
    pcts = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        normal = rng.normal(0, 5, 500)
        events = rng.choice(normal, 20, replace=False)
        pcts.append(placebo_percentile(events, normal))
    pcts = np.array(pcts)

    assert 35 < np.median(pcts) < 65, f"biased: median {np.median(pcts):.1f}"
    assert pcts.std() > 18, (
        f"percentiles clustered (sd {pcts.std():.1f}) -- the reference "
        "distribution is too wide, probably built from single days")
    assert pcts.min() < 15 and pcts.max() > 85


def test_placebo_detects_a_real_drop():
    """A genuine fall must land in the low tail."""
    normal = np.random.default_rng(4).normal(0, 5, 500)
    events = np.full(20, -12.0)
    assert placebo_percentile(events, normal) < 5


def test_placebo_detects_a_real_rise():
    """And a genuine spike in the high tail -- the June 2024 election case."""
    normal = np.random.default_rng(5).normal(0, 5, 500)
    events = np.full(10, +20.0)
    assert placebo_percentile(events, normal) > 95


def test_placebo_compares_means_to_means_not_to_singles():
    """The trap, tested directly.

    Twenty-three events averaging -2.0 against a 10% daily spread IS a real
    signal: the average of 23 ordinary days has a spread of only 10/sqrt(23),
    about 2.1, so -2.0 is nearly a full standard error below zero and lands
    around the 15th percentile.

    Judged against single days the reference is sqrt(23) times too wide and
    the same -2.0 reads as roughly the 40th percentile -- unremarkable. That
    is the bug, and it is conservative: it buries findings rather than
    manufacturing them. This test fails if the reference is built wrongly.
    """
    normal = np.random.default_rng(6).normal(0, 10, 2000)
    events = np.full(23, -2.0)
    pct = placebo_percentile(events, normal)
    assert pct < 25, (
        f"got {pct:.1f} -- expected roughly 15. A reference built from single "
        "days rather than means of n would give about 40 and miss the effect")


def test_placebo_sample_size_matters():
    """The same observed mean is more convincing from more events.

    Three events averaging -8 could easily be luck; forty averaging -8 could
    not. A correct implementation ranks the larger sample as more extreme."""
    normal = np.random.default_rng(7).normal(0, 6, 1000)
    few = placebo_percentile(np.full(3, -8.0), normal)
    many = placebo_percentile(np.full(40, -8.0), normal)
    assert many < few


def test_placebo_in_range_and_guarded():
    normal = np.random.default_rng(8).normal(0, 5, 300)
    assert 0.0 <= placebo_percentile(np.full(5, -3.0), normal) <= 100.0
    assert math.isnan(placebo_percentile([], normal))
    assert math.isnan(placebo_percentile([np.nan, np.nan], normal))
    assert math.isnan(placebo_percentile([-5.0], [1.0, 2.0, 3.0]))   # pool < 30


def test_placebo_is_reproducible():
    normal = list(np.random.default_rng(9).normal(0, 5, 200))
    events = [-6.0, -7.0, -5.0]
    assert (placebo_percentile(events, normal, seed=11)
            == placebo_percentile(events, normal, seed=11))


# ===========================================================================
# expected_move_pct
# ===========================================================================

def test_expected_move_basic_arithmetic():
    """A 300 + 280 straddle on a 24,000 index is 580/24000 = 2.4167%."""
    assert expected_move_pct(300.0, 280.0, 24000.0) == pytest.approx(2.41666, abs=1e-4)


def test_expected_move_scales_with_price_not_level():
    """Doubling the straddle doubles the expected move."""
    one = expected_move_pct(200.0, 200.0, 20000.0)
    two = expected_move_pct(400.0, 400.0, 20000.0)
    assert two == pytest.approx(2 * one)


def test_expected_move_is_relative_to_spot():
    """The same straddle on a higher index implies a smaller percentage."""
    assert expected_move_pct(500.0, 500.0, 25000.0) < expected_move_pct(
        500.0, 500.0, 20000.0)


def test_expected_move_guards_bad_input():
    assert math.isnan(expected_move_pct(300.0, 280.0, 0.0))
    assert math.isnan(expected_move_pct(300.0, 280.0, -24000.0))
    assert math.isnan(expected_move_pct(np.nan, 280.0, 24000.0))
    assert math.isnan(expected_move_pct(300.0, np.nan, 24000.0))


# ===========================================================================
# Given helpers -- these already pass
# ===========================================================================

def test_one_day_changes():
    out = one_day_changes([100.0, 110.0, 99.0])
    assert len(out) == 2
    assert out[0] == pytest.approx(10.0)
    assert out[1] == pytest.approx(-10.0)


def test_normal_day_mask_excludes_the_window():
    import datetime as dt
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(50)]
    mask = normal_day_mask(days, [days[25]], gap=10)
    assert not mask[25]                       # the event day
    assert not mask[15] and not mask[35]      # window edges
    assert mask[0] and mask[49]               # far away
    assert mask.sum() == 50 - 21


def test_skew_sign_and_nan():
    assert skew(16.0, 13.0) == pytest.approx(3.0)
    assert skew(12.0, 14.0) == pytest.approx(-2.0)
    assert math.isnan(skew(np.nan, 13.0))
    assert math.isnan(skew(16.0, np.nan))
