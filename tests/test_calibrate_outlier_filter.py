"""Probes for the tail-gap cut and the bounded accumulator.

The rule exists so that outlier removal needs no per-variable clinical bounds. What it must do
is cut where a distribution detaches -- errors sit orders of magnitude off the body -- and leave
a continuous tail alone however far that tail runs, because a lactate of 28 and a glucose of
2000 are real.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.calibrate_outlier_filter import (
    Accumulator, EXTREMES, RESERVOIR, tail_gap_cut)


DEFAULTS = dict(gap=0.5, quantile=0.999, min_fold=3.0)


def cut(values):
    return tail_gap_cut(np.asarray(values, dtype=float), **DEFAULTS)


def test_a_detached_upper_cluster_is_cut_off():
    rng = np.random.default_rng(0)
    body = rng.normal(37.0, 0.6, 20_000)
    errors = np.array([2_575.0, 185_000.0])
    low, high = cut(np.concatenate([body, errors]))
    # Strictly inside the gap, so the cluster is excluded and the body is kept whichever way
    # the comparison rounds.
    assert body.max() < high < 2_575.0, f'cut at {high} is not inside the gap'


def test_a_continuous_tail_is_left_alone_however_far_it_runs():
    """Lactate runs from 0.4 to 28 with no break. Nothing in it is an error."""
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.lognormal(np.log(1.6), 0.5, 40_000),
                             rng.uniform(15, 28, 200)])
    low, high = cut(values)
    assert high == np.inf, f'cut a continuous tail at {high}'


def test_the_guard_stops_a_cut_near_the_median():
    """A sparse but genuine tail must not be sliced just because it has gaps."""
    values = np.concatenate([np.full(5_000, 10.0), np.array([12.0, 15.0, 20.0])])
    low, high = cut(values)
    assert high == np.inf or high >= 3.0 * 10.0


def test_a_detached_lower_cluster_is_cut_off():
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(96.0, 3.0, 20_000).clip(60, 100),
                             np.array([0.97, 0.88])])
    low, high = cut(values)
    assert 0.97 < low < 60.0, f'lower cut at {low} did not isolate the 0-1 scale entries'


def test_too_few_values_cuts_nothing():
    """A rare assay has no distribution to speak of; the rule must abstain, not guess."""
    low, high = cut([1.0, 2.0, 900.0])
    assert (low, high) == (-np.inf, np.inf)


def test_non_positive_values_do_not_break_the_log():
    """Base excess and anion gap are legitimately negative."""
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(0.0, 3.0, 5_000), np.array([-40.0, 50.0])])
    low, high = cut(values)
    assert np.isfinite(low) or low == -np.inf
    assert np.isfinite(high) or high == np.inf


def test_a_genuinely_bimodal_variable_is_not_cut_at_the_valley():
    """Two dense real populations are not an error, however far apart they sit."""
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.lognormal(np.log(0.5), 0.3, 10_000),
                             rng.lognormal(np.log(80.0), 0.3, 10_000)])
    low, high = cut(values)
    assert high == np.inf, f'cut a bimodal distribution at {high}'


class TestAccumulator:
    def test_the_reservoir_is_capped(self):
        accumulator = Accumulator()
        for _ in range(3):
            accumulator.add(np.arange(RESERVOIR))
        assert len(accumulator.reservoir) == RESERVOIR
        assert accumulator.count == 3 * RESERVOIR

    def test_both_extremes_survive_regardless_of_order(self):
        """The tails are what the gap search walks, so they cannot be lost to sampling."""
        accumulator = Accumulator()
        accumulator.add(np.array([1e9]))
        for _ in range(20):
            accumulator.add(np.random.default_rng(0).normal(10, 1, 10_000))
        accumulator.add(np.array([1e-9]))
        sample = accumulator.tail_sample()
        assert sample.max() == pytest.approx(1e9)
        assert sample.min() == pytest.approx(1e-9)

    def test_removals_are_counted_once(self):
        """The reservoir and the extremes overlap while the reservoir is not full, so counting
        over their concatenation reports a single error two or three times."""
        accumulator = Accumulator()
        accumulator.add(np.concatenate([np.full(5_000, 10.0), [1e6]]))
        removed, saturated = accumulator.beyond(-np.inf, 1e5)
        assert removed == 1
        assert not saturated

    def test_saturation_is_reported(self):
        """If more values lie beyond a cut than the tails retain, the count is a floor and the
        rule is removing far too much to adopt."""
        accumulator = Accumulator()
        accumulator.add(np.full(EXTREMES + 10, 1e6))
        removed, saturated = accumulator.beyond(-np.inf, 1e5)
        assert saturated
