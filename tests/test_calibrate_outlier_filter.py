"""Probes for the tail-gap cut and the bounded accumulator.

The rule exists so that outlier removal needs no per-variable clinical bounds. What it must do
is cut where a distribution detaches -- errors sit orders of magnitude off the body -- and leave
a continuous tail alone however far that tail runs, because a lactate of 28 and a glucose of
2000 are real.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.calibrate_outlier_filter import (
    Accumulator, EXTREMES, RESERVOIR, UNDECLARED, declared_unit, numeric_values,
    print_reading_guide, report_unit_audit, robust_log_cut, robust_log_z,
    tail_gap_cut)


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


class TestAccumulatorScales:
    """The tool reads a hundred thousand subjects' worth of events on a shared node, so the
    accumulators have to be bounded in memory and linear in time."""

    def test_the_reservoir_is_a_uniform_sample_once_full(self):
        """Reservoir sampling is only correct if late values can displace early ones."""
        accumulator = Accumulator(reservoir=1_000, extremes=100)
        rng = np.random.default_rng(0)
        for _ in range(50):
            accumulator.add(rng.normal(0.0, 1.0, 1_000))
        assert accumulator.count == 50_000
        assert accumulator.reservoir.size == 1_000
        # A sample of a standard normal, not the first thousand values only.
        assert abs(float(np.mean(accumulator.reservoir))) < 0.15
        assert 0.85 < float(np.std(accumulator.reservoir)) < 1.15

    def test_the_tails_are_exact_not_sampled(self):
        """The gap search walks the tail, so an error must not be lost to reservoir sampling."""
        accumulator = Accumulator(reservoir=1_000, extremes=100)
        rng = np.random.default_rng(1)
        for _ in range(50):
            accumulator.add(rng.normal(10.0, 1.0, 1_000))
        accumulator.add(np.array([1e7]))
        for _ in range(50):
            accumulator.add(rng.normal(10.0, 1.0, 1_000))
        assert accumulator.high.max() == pytest.approx(1e7), 'the outlier was sampled away'

    def test_memory_stays_bounded(self):
        accumulator = Accumulator(reservoir=1_000, extremes=100)
        rng = np.random.default_rng(2)
        for _ in range(200):
            accumulator.add(rng.normal(0.0, 1.0, 1_000))
        assert accumulator.reservoir.size == 1_000
        assert accumulator.high.size == 100
        assert accumulator.low.size == 100
        assert accumulator.count == 200_000

    def test_pending_values_are_not_lost_before_a_trim(self):
        """Values added since the last trim must still be visible to the tails."""
        accumulator = Accumulator(reservoir=10, extremes=1_000)
        accumulator.add(np.array([5.0, 6.0]))
        accumulator.add(np.array([1e6]))
        assert accumulator.high.max() == pytest.approx(1e6)


def frame(rows):
    return pd.DataFrame(rows, columns=['VARIABLE', 'ITEMID', 'VALUE'])


def test_a_text_variable_is_dropped_rather_than_raising():
    kept = numeric_values(frame([
        ('Glucose', 220621, '5.4'),
        ('Discharge note', 900001, 'patient stable overnight'),
        ('Discharge note', 900001, ''),
    ]))
    assert list(kept['VARIABLE']) == ['Glucose']
    assert kept['VALUE'].to_numpy(dtype=float) == pytest.approx([5.4])


def test_free_text_in_a_numeric_variable_drops_only_that_row():
    kept = numeric_values(frame([
        ('Glucose', 220621, '5.4'),
        ('Glucose', 220621, 'unable to obtain'),
        ('Glucose', 220621, 7.1),
    ]))
    assert kept['VALUE'].to_numpy(dtype=float) == pytest.approx([5.4, 7.1])


def test_missing_and_non_finite_values_are_dropped():
    kept = numeric_values(frame([
        ('Glucose', 220621, np.nan),
        ('Glucose', 220621, np.inf),
        ('Glucose', 220621, '6.0'),
    ]))
    assert kept['VALUE'].to_numpy(dtype=float) == pytest.approx([6.0])


def loaded(values):
    accumulator = Accumulator()
    accumulator.add(np.asarray(values, dtype=float))
    return accumulator


def test_surviving_reports_the_observed_range_when_nothing_is_cut():
    accumulator = loaded([1.0, 5.0, 50.0, 900.0])
    assert accumulator.surviving(-np.inf, np.inf) == (1.0, 900.0)


def test_surviving_stops_at_the_last_value_inside_the_cuts():
    accumulator = loaded([0.1, 1.0, 5.0, 50.0, 900.0, 90000.0])
    smallest, largest = accumulator.surviving(0.5, 5000.0)
    assert (smallest, largest) == (1.0, 900.0)


def test_surviving_is_nan_when_a_cut_excludes_every_retained_value():
    accumulator = loaded([100.0, 200.0, 300.0])
    smallest, largest = accumulator.surviving(-np.inf, 1.0)
    assert smallest == 100.0
    assert np.isnan(largest)


@pytest.mark.parametrize('given', [None, '', '   ', float('nan'), 'nan', 'None'])
def test_a_blank_unitname_is_undeclared(given):
    assert declared_unit(given) == UNDECLARED


def test_unitnames_compare_ignoring_case_and_padding():
    assert declared_unit(' mL ') == declared_unit('ml')


def unit_map(rows):
    return pd.DataFrame(
        [(itemid, variable, label, unit) for variable, itemid, label, unit in rows],
        columns=['ITEMID', 'VARIABLE', 'LABEL', 'UNITNAME']).set_index('ITEMID')


def audit(rows, medians, warn_ratio=1.5):
    by_itemid = {}
    for itemid, centre in medians.items():
        variable = next(r[0] for r in rows if r[1] == itemid)
        by_itemid[(variable, itemid)] = loaded(np.full(500, float(centre)))
    import argparse
    report_unit_audit(by_itemid, unit_map(rows), argparse.Namespace(warn_ratio=warn_ratio))


URINE = [
    ('Urine output', 226559, 'Foley', 'mL'),
    ('Urine output', 226631, 'PACU Urine', 'mL'),
    ('Urine output', 226567, 'Straight Cath', 'mL'),
]
WEIGHT = [
    ('Weight', 224639, 'Daily Weight', 'kg'),
    ('Weight', 226531, 'Admission Weight (lbs.)', 'lb'),
]


def test_one_unit_is_never_flagged_however_far_the_medians_diverge(capsys):
    audit(URINE, {226559: 100, 226631: 700, 226567: 500})
    out = capsys.readouterr().out
    assert 'No variable pools itemids that declare different units.' in out
    assert 'Urine output' not in out


def test_mixed_units_with_agreeing_medians_read_as_converted(capsys):
    audit(WEIGHT, {224639: 79.8, 226531: 80.1})
    out = capsys.readouterr().out
    assert 'Weight' in out
    assert 'the conversion is in place' in out


def test_mixed_units_with_diverging_medians_read_as_missing(capsys):
    audit(WEIGHT, {224639: 79.8, 226531: 176.0})
    assert 'the conversion is MISSING' in capsys.readouterr().out


def test_a_unit_observed_alone_is_not_judged(capsys):
    audit(WEIGHT, {224639: 79.8})
    out = capsys.readouterr().out
    assert 'only one of these units was observed' in out
    assert 'conversion is' not in out


def test_an_itemid_with_no_declared_unit_is_reported_separately(capsys):
    rows = WEIGHT + [('Weight', 226512, 'Admission Weight', None)]
    audit(rows, {224639: 79.8, 226531: 80.1, 226512: 80.0})
    out = capsys.readouterr().out
    assert 'declare a unit for some itemids and none for others' in out
    assert '226512' in out


def spread_cut(values, k=8.0):
    return robust_log_cut(np.asarray(values, dtype=float), k)


def test_the_threshold_scales_to_the_variables_own_spread():
    """The same k must mean something different inoriginal units for a tight and a wide variable."""
    rng = np.random.default_rng(0)
    tight = rng.normal(36.9, 0.7, 20000)
    wide = rng.lognormal(np.log(30), 1.0, 20000)
    _, tight_high, _ = spread_cut(tight)
    _, wide_high, _ = spread_cut(wide)
    assert tight_high < 60          # a temperature of 60 is not a temperature
    assert wide_high > 5000         # an ALT of 5000 is an acute liver injury


def test_a_continuous_heavy_tail_survives_where_a_tight_variable_is_cut():
    """Both values sit the same number of decades out; only the spread tells them apart."""
    rng = np.random.default_rng(1)
    tight = np.concatenate([rng.normal(36.9, 0.7, 20000), [530.6]])
    wide = np.concatenate([rng.lognormal(np.log(30), 1.0, 20000), [13960.0]])
    assert spread_cut(tight)[1] < 530.6
    assert spread_cut(wide)[1] > 13960.0


def test_a_bridging_value_does_not_save_an_outlier_from_the_spread_rule():
    """The case the gap rule loses: one value between the body and a far one."""
    rng = np.random.default_rng(2)
    bridged = np.concatenate([rng.normal(36.9, 0.7, 20000), [110.0, 250.0, 530.6]])
    assert tail_gap_cut(bridged, gap=0.5, quantile=0.999, min_fold=3.0)[1] == np.inf
    assert spread_cut(bridged)[1] < 110.0


def test_too_few_values_cuts_nothing_and_reports_no_spread():
    low, high, scale = spread_cut(np.full(50, 5.0))
    assert (low, high) == (-np.inf, np.inf)
    assert np.isnan(scale)


def test_a_variable_with_no_spread_at_all_cuts_nothing():
    low, high, scale = spread_cut(np.full(500, 5.0))
    assert (low, high) == (-np.inf, np.inf)
    assert scale == 0.0


def test_z_is_undefined_where_the_log_rules_cannot_look():
    assert np.isnan(robust_log_z(0.0, 1.5, 0.01))
    assert np.isnan(robust_log_z(-17.78, 1.5, 0.01))
    assert np.isnan(robust_log_z(100.0, 1.5, 0.0))


def test_zeros_and_negatives_are_counted_but_not_charged_to_the_cut():
    accumulator = loaded(np.concatenate([np.full(900, 37.0), np.zeros(60), np.full(40, -17.78)]))
    assert (accumulator.zeros, accumulator.negatives) == (60, 40)
    assert accumulator.positive_count == 900
    removed, _ = accumulator.beyond(30.0, 45.0)
    assert removed == 0


def guide_lines(**overrides):
    import argparse
    settings = dict(gap=0.5, min_fold=3.0, z=8.0, warn_fraction=0.001)
    settings.update(overrides)
    import io as _io, contextlib
    buffer = _io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_reading_guide(argparse.Namespace(**settings))
    return buffer.getvalue().splitlines()


def test_the_guide_fits_the_width_of_the_tables():
    assert max(len(line) for line in guide_lines()) <= 100


def test_the_guide_states_the_settings_the_run_actually_used():
    text = '\n'.join(guide_lines(gap=0.75, min_fold=5.0, z=11.0))
    assert 'currently 0.75' in text
    assert 'currently 5.0' in text
    assert 'currently 11.0' in text


def test_the_guide_survives_a_warn_fraction_of_a_different_width():
    assert max(len(line) for line in guide_lines(warn_fraction=0.025)) <= 100
