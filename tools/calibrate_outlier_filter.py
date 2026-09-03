"""What an algorithmic outlier filter would remove, measured against the real data.

`variable_ranges.csv` is a hand-maintained table with no generator in either this repository or
upstream, and it is never read -- `extract_episodes_from_subjects.py` declares
`--reference_range_file` and does nothing with it. Curating clinical bounds per variable does
not scale to a hundred lab tests, let alone a thousand, so this measures two rules that need no
per-variable knowledge:

    tail gap    Sort the log values and cut at the first jump wider than `--gap` decades. A
                genuine tail is continuous however far it extends; errors from decimal slips,
                unit slips and sentinels sit in a separate cluster orders of magnitude out. The
                cut point adapts per variable, so nothing has to be told how far a real tail
                runs. Its weakness is that one erroneous value between the body and a far one
                bridges the gap that was supposed to separate them.

    robust      Cut at `--z` robust spreads from the median, in log space, with the spread
    spread      measured from the variable itself. Nothing bridges anything, and the threshold
                means the same thing for a variable whose body spans a factor of two as for
                one that spans three orders of magnitude.

    unit audit  Read the declared UNITNAME of every ITEMID pooled under one VARIABLE. An
                ITEMID recorded in a unit nothing converts is invisible per value -- every
                reading is a plausible number -- but it shifts the whole distribution. The map
                declares the unit, so this needs no threshold; the observed medians are then
                used only to say whether the conversion actually ran.

Neither rule is applied here. This reports what they would do, so the thresholds can be set from
evidence before anything is wired into `cleaners.py`.

No stage of the pipeline is re-run: `events.csv` is the output of step 1, already on disk, and
the only work done to it is the in-memory mapping and cleaning that step 3 performs before the
filter would sit. The raw MIMIC-IV tables are never touched.

Usage:
    python tools/calibrate_outlier_filter.py <subjects_root> \\
        --variable_map_file mimic4dataprep/resources/itemid_to_variable_map_used.csv \\
        --subjects 2000
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

from mimic4dataprep.cleaners import clean_events
from mimic4dataprep.preprocessing import map_itemids_to_variables, read_itemid_to_variable_map
from mimic4dataprep.subject import read_events


# Values kept per variable. The reservoir carries the body of the distribution for quantiles;
# the extremes carry the tail the gap search walks. Both are capped, so memory does not grow
# with the number of subjects read.
RESERVOIR = 200_000
EXTREMES = 20_000

# An itemid accumulator only ever yields a median, so it needs a fraction of the sample. With a
# few hundred itemids, giving each the full variable-sized reservoir would cost gigabytes.
ITEMID_RESERVOIR = 20_000
ITEMID_EXTREMES = 1_000


class Accumulator:
    """Bounded sample of one series: a reservoir for quantiles, plus both tails in full.

    Backed by numpy rather than Python lists -- a float in a list costs 32 bytes against 8 --
    and the tails are trimmed lazily. Trimming on every call would re-scan the retained tail
    once per subject per variable, which is quadratic in the wrong quantity.
    """

    def __init__(self, reservoir=RESERVOIR, extremes=EXTREMES):
        self.count = 0
        # Neither log rule can see these: log10 is undefined at zero and below, so both
        # drop them before searching and neither can report on them.
        self.zeros = 0
        self.negatives = 0
        self.capacity = reservoir
        self.extremes = extremes
        self._reservoir = np.empty(reservoir, dtype=np.float64)
        self._filled = 0
        self._high = np.empty(0, dtype=np.float64)
        self._low = np.empty(0, dtype=np.float64)
        self._pending = []

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return

        # Reservoir sampling: every value seen keeps an equal chance of being retained.
        take = min(self.capacity - self._filled, values.size)
        if take:
            self._reservoir[self._filled:self._filled + take] = values[:take]
            self._filled += take
        for offset in range(take, values.size):
            j = random.randrange(self.count + offset + 1)
            if j < self.capacity:
                self._reservoir[j] = values[offset]
        self.count += values.size
        self.zeros += int((values == 0).sum())
        self.negatives += int((values < 0).sum())

        self._pending.append(values)
        if sum(p.size for p in self._pending) >= self.extremes:
            self._trim()

    def _trim(self):
        """Fold the pending values into the retained tails. O(n) via partition, not a sort."""
        if not self._pending:
            return
        combined = np.concatenate([self._high, self._low] + self._pending)
        self._pending = []
        k = min(self.extremes, combined.size)
        self._high = np.partition(combined, -k)[-k:]
        self._low = np.partition(combined, k - 1)[:k]

    @property
    def high(self):
        self._trim()
        return self._high

    @property
    def low(self):
        self._trim()
        return self._low

    @property
    def reservoir(self):
        return self._reservoir[:self._filled]

    def tail_sample(self):
        """Reservoir plus both tails, for the gap search.

        The two overlap while the reservoir is not yet full, so this is not a sample of the
        distribution and must not be counted against. Finding a gap does not care.
        """
        return np.concatenate([self.reservoir, self.high, self.low])

    def body_sample(self):
        """The reservoir alone: a uniform sample of every value seen, for quantiles."""
        return self.reservoir

    def surviving(self, low, high):
        """The smallest and largest values the cuts would keep, in original units.

        The cuts alone do not say how much room is left between them and the data -- a high
        cut of 4500 on a variable whose largest real value is 22 is a different proposition
        from one whose largest is 4000. Read off the retained tails, so exact under the same
        condition as `beyond`, and NaN on the side where saturation put the answer outside
        the retained values.
        """
        kept_low = self.low[self.low >= low]
        kept_high = self.high[self.high <= high]
        return (float(kept_low.min()) if kept_low.size else np.nan,
                float(kept_high.max()) if kept_high.size else np.nan)

    @property
    def positive_count(self):
        """Observations both log rules could actually see."""
        return self.count - self.zeros - self.negatives

    def beyond(self, low, high, positive_only=True):
        """How many observations fall outside the cuts.

        Counted from the retained extremes rather than the reservoir, so it is exact rather
        than sampled -- unless more than EXTREMES values lie beyond a cut, which would mean
        the rule is removing far too much to adopt anyway.

        With `positive_only`, zeros and negatives are excluded: the log rules derive their
        cuts from positive values alone, so counting values they never saw against them
        reads a variable's sentinel share as if the rule had chosen to remove it. A rule
        that works in original units sees everything and passes False.
        """
        tail_high, tail_low = self.high, self.low
        if positive_only:
            tail_high, tail_low = tail_high[tail_high > 0], tail_low[tail_low > 0]
        above = int((tail_high > high).sum())
        below = int((tail_low < low).sum())
        saturated = (above >= tail_high.size == self.extremes
                     or below >= tail_low.size == self.extremes)
        return above + below, saturated


def tail_gap_cut(values, gap, quantile, min_fold):
    """Where the upper and lower tails detach from the body, in original units.

    Args:
        values: Observed values for one variable.
        gap: Width in decades of the jump that counts as a detachment.
        quantile: Only search beyond this quantile, so the body is never cut.
        min_fold: Never cut closer to the median than this multiple, whatever the gap search
            finds. A guard against slicing into a variable whose tail is genuinely sparse.

    Returns:
        (low_cut, high_cut), either bound infinite if no detachment was found. Each cut is
        placed in the middle of the gap rather than on the first detached value: a value round
        tripped through log10 and back does not always compare equal to itself, so a cut set on
        the datum can fail to exclude it.
    """
    positive = values[values > 0]
    if positive.size < 100:
        return -np.inf, np.inf

    y = np.sort(np.log10(positive))
    median = float(np.median(y))
    guard = np.log10(min_fold)

    high_cut = np.inf
    start = np.searchsorted(y, max(float(np.quantile(y, quantile)), median + guard))
    for i in range(max(start, 1), y.size):
        if y[i] - y[i - 1] > gap:
            high_cut = 10.0 ** ((y[i] + y[i - 1]) / 2.0)
            break

    low_cut = -np.inf
    stop = np.searchsorted(y, min(float(np.quantile(y, 1 - quantile)), median - guard))
    for i in range(min(stop, y.size - 1), 0, -1):
        if y[i] - y[i - 1] > gap:
            low_cut = 10.0 ** ((y[i] + y[i - 1]) / 2.0)
            break

    return low_cut, high_cut


def numeric_values(events):
    """The VARIABLE/ITEMID/VALUE rows that carry a usable number.

    Coercion is per value, not per column: text variables and free-text entries in an
    otherwise numeric variable become NaN and drop out here. A whole-column cast raises on
    the first string, and pandas' errors='ignore' would leave the column untouched,
    deferring the same failure to the first arithmetic on it.
    """
    numeric = events[['VARIABLE', 'ITEMID', 'VALUE']].copy()
    numeric['VALUE'] = pd.to_numeric(numeric['VALUE'], errors='coerce')
    return numeric[np.isfinite(numeric['VALUE'].to_numpy(dtype=float))]


def read_subject(subject_path, var_map):
    """The cleaned events for one subject, exactly as step 3 would produce them."""
    events = read_events(subject_path)
    events = map_itemids_to_variables(events, var_map)
    if events.shape[0] == 0:
        return None
    events = clean_events(events, var_map)
    return events if events.shape[0] else None


def collect(subjects_root, var_map, n_subjects, seed):
    """Accumulate per-variable and per-itemid samples over a sample of subjects."""
    directories = sorted(
        d for d in os.listdir(subjects_root)
        if os.path.isdir(os.path.join(subjects_root, d))
        and os.path.exists(os.path.join(subjects_root, d, 'events.csv'))
    )
    rng = random.Random(seed)
    if n_subjects and n_subjects < len(directories):
        directories = rng.sample(directories, n_subjects)
    print(f'reading {len(directories):,} subjects from {subjects_root}', flush=True)

    by_variable, by_itemid, failures, seen = {}, {}, 0, set()
    for index, name in enumerate(directories, 1):
        try:
            events = read_subject(os.path.join(subjects_root, name), var_map)
        except Exception:
            failures += 1
            continue
        if events is None:
            continue
        seen.update(events['VARIABLE'].unique())
        numeric = numeric_values(events)
        for variable, group in numeric.groupby('VARIABLE'):
            by_variable.setdefault(variable, Accumulator()).add(
                group['VALUE'].to_numpy(dtype=float))
            for itemid, sub in group.groupby('ITEMID'):
                by_itemid.setdefault(
                    (variable, itemid),
                    Accumulator(ITEMID_RESERVOIR, ITEMID_EXTREMES)).add(
                        sub['VALUE'].to_numpy(dtype=float))
        if index % 500 == 0:
            print(f'  {index:,} subjects, {len(by_variable)} variables', flush=True)

    if failures:
        print(f'  {failures:,} subjects could not be read and were skipped')
    non_numeric = sorted(seen - set(by_variable))
    if non_numeric:
        print(f'  {len(non_numeric)} variables carried no numeric values and are not '
              f'reported: {", ".join(non_numeric)}')
    return by_variable, by_itemid


def robust_log_cut(values, k):
    """Cut at a fixed number of robust spreads from the median, in log space.

    The gap rule fails on two counts that no threshold of its own can fix: a single
    erroneous value between the body and a far one bridges the gap that was supposed to
    separate them, and its guard is a fixed multiple of the median, which is many spreads
    out for a tight variable and inside the normal range for a wide one.

    Working in log space makes the rule scale-free, and taking the spread from the variable
    itself makes the threshold mean the same thing everywhere: a temperature 1.2 decades out
    is absurd, an ALT 2.7 decades out is a real acute liver injury, and the two are told
    apart by how far their own bodies spread, not by a decade count.

    Returns:
        (low_cut, high_cut, scale), where scale is one robust spread in decades. Infinite
        cuts and a NaN scale when there is too little to measure, or a zero scale when the
        variable is so discrete that half its values sit on the median.
    """
    positive = values[values > 0]
    if positive.size < 100:
        return -np.inf, np.inf, np.nan

    y = np.log10(positive)
    centre = float(np.median(y))
    # 1.4826 puts the MAD on the same footing as a standard deviation for normal data, so k
    # reads as a number of sigmas rather than an arbitrary unit.
    scale = 1.4826 * float(np.median(np.abs(y - centre)))
    if scale <= 0:
        return -np.inf, np.inf, 0.0
    return 10.0 ** (centre - k * scale), 10.0 ** (centre + k * scale), scale


def robust_log_z(value, centre, scale):
    """How many robust spreads a value sits from the median, in log space."""
    if not np.isfinite(value) or value <= 0 or not scale or not np.isfinite(scale):
        return np.nan
    return (np.log10(value) - centre) / scale


TAIL_GAP_WIDTH = 124


def _number(value):
    """A table cell for a value that may not exist, when a saturated tail hid it."""
    return 'n/a' if not np.isfinite(value) else f'{value:.4g}'


def report_tail_gap(by_variable, args):
    """Per variable, where the rule would cut and how much it would remove."""
    print(f'\n{"=" * TAIL_GAP_WIDTH}')
    print(f'TAIL GAP  (gap > {args.gap} decades, beyond q{args.quantile}, '
          f'no closer than {args.min_fold}x the median)')
    print('=' * TAIL_GAP_WIDTH)
    print(f'{"variable":<34}{"observed":>11}{"median":>10}{"low cut":>12}{"min kept":>12}'
          f'{"max kept":>13}{"high cut":>13}{"removed":>10}{"%":>9}')
    print('-' * TAIL_GAP_WIDTH)

    for variable in sorted(by_variable):
        accumulator = by_variable[variable]
        low, high = tail_gap_cut(accumulator.tail_sample(),
                                 args.gap, args.quantile, args.min_fold)
        removed, saturated = accumulator.beyond(low, high)
        seen = accumulator.positive_count
        share = removed / seen if seen else 0.0
        marker = '  <-- inspect' if share > args.warn_fraction else ''
        if saturated:
            marker = '  <-- SATURATED, removing far too much'
        smallest, largest = accumulator.surviving(low, high)
        print(f'{variable[:33]:<34}{accumulator.count:>11,}'
              f'{np.median(accumulator.body_sample()):>10.3g}'
              f'{("none" if low == -np.inf else f"{low:.4g}"):>12}'
              f'{_number(smallest):>12}{_number(largest):>13}'
              f'{("none" if high == np.inf else f"{high:.4g}"):>13}'
              f'{removed:>10,}{share:>9.4%}{marker}')
    print('-' * TAIL_GAP_WIDTH)
    print(f'  Counts are exact while a tail holds fewer than its cap of retained values.')
    print(f'  Anything above {args.warn_fraction:.2%} is removing more than errors and '
          f'wants looking at before the rule is adopted.')


RANGE_COLUMNS = ('VARIABLE', 'MEDIAN', 'SCALE', 'LOW', 'HIGH', 'U')


def emit_ranges(by_variable, args, path):
    """Write the standardized-magnitude cuts for cleaners.py to apply.

    Extraction runs one subject at a time and so has no view of a variable's distribution.
    The cuts are therefore measured once, here, over a sample of subjects, and written out
    for the per-subject pass to read. Percentiles and a median are stable at this sample
    size and are not disturbed by the values being removed, so the file does not need
    regenerating as the data changes.
    """
    lines = [','.join(RANGE_COLUMNS)]
    written = 0
    for variable in sorted(by_variable):
        centre, scale = standardized_scale(by_variable[variable].body_sample())
        if not np.isfinite(scale):
            continue
        low, high = centre - args.u * scale, centre + args.u * scale
        lines.append(f'{variable},{centre:.10g},{scale:.10g},{low:.10g},{high:.10g},{args.u:g}')
        written += 1
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(f'\nwrote {written} variable ranges at --u {args.u:g} to {path}')
    skipped = sorted(set(by_variable) - {line.split(',')[0] for line in lines[1:]})
    if skipped:
        print(f'  {len(skipped)} variable(s) have no usable percentile range and are not '
              f'filtered: {", ".join(skipped)}')


def print_reading_guide(args):
    """What the report measures and how to act on it, ahead of the tables themselves."""
    print(f'{"=" * 100}')
    print('HOW TO READ THIS')
    print('=' * 100)
    print(f"""
Nothing here is applied to any data. Two candidate outlier rules are measured against the
extracted events, and this reports what each would remove, so a threshold can be chosen from
evidence before anything is wired into cleaners.py.

THE TWO RULES

  tail gap    Sorts the log values and cuts at the first jump wider than --gap decades
              (currently {args.gap}). A real tail is continuous however far it runs, while errors
              from decimal slips, unit slips and sentinels sit in a separate cluster orders of
              magnitude out. Two weaknesses: one erroneous value lying between the body and a
              far one bridges the gap that was supposed to separate them, and the --min_fold
              guard (currently {args.min_fold}) is a fixed multiple of the median, which for
              a tight variable floors the search above the values it needed to reach.

  robust      Cuts at --z robust spreads from the median, in log space (currently {args.z}).
  spread      Nothing bridges anything, and the threshold scales to each variable. Its
              weakness is what it measures: a MAD spans the regulated core of a variable, not
              its clinically observed range. Sodium is held to 135-145, so a real 175 reads as
              hundreds of spreads, while a variable that ranges over orders of magnitude in
              ordinary illness would hide a gross error in a handful.

  standard-   Cuts beyond --u units of (p95 - p5) from the median (currently {args.u}), the
  ized        scale standardize_feats divides by. Aimed at a different question: not whether a
  magnitude   value is clinically possible, but how much of the loss it takes. A value at u
              costs u^2, so this bounds the damage rather than judging the value.

WHICH QUESTION IS BEING ASKED

  Two objectives are easy to conflate and want different rules:

      (a) remove clinically impossible values
      (b) stop extreme values from dominating the loss

  (a) is not reachable by a threshold on any single statistic. Real pathology and gross error
  overlap on every scale-free measure, because how far a variable's plausible range extends
  beyond its usual one is a fact about physiology, not about its distribution.

  (b) is reachable, and directly measurable, because the harm is not that a value is wrong but
  that it is large after standardization. A height of 419cm is impossible and costs a loss
  almost nothing; a temperature of 185000 costs it everything. Those want different responses,
  and only the standardized magnitude tells them apart.

HOW THE ROBUST SPREAD RULE WORKS

  For one variable, over its positive values v, with y = log10(v):

      m    = median(y)                      the centre, in decades
      s    = 1.4826 * median(|y - m|)       one robust spread, in decades
      z(v) = (log10(v) - m) / s             distance from the centre, in spreads

  The rule keeps |z| <= k, which in original units is a cut at 10^(m +/- k*s).

  s is measured per variable, so a single k means something different for each. Illustrative
  figures for a tight variable and a wide one, both at k = 8:

      Temperature   median 36.9   s = 0.008 decades   one spread = 1.02x   ->  cut at 42.9
      ALT           median 29.9   s = 0.433 decades   one spread = 2.71x   ->  cut at 87000

  Nothing was told that 43 degrees is implausible or that 87000 U/L is reachable. Both follow
  from the spread of each variable's own body.

  A median absolute deviation is used rather than a standard deviation because the values being
  looked for would inflate an SD and so help hide themselves. The 1.4826 rescales the MAD to
  match a standard deviation on normal data, so k reads as a number of sigmas.

CHOOSING --z OR --u

  z(min) and z(max) in the ROBUST LOG SPREAD table, and u(min) and u(max) in the STANDARDIZED
  MAGNITUDE table, are measured on the observed values before any cut. They do not depend on
  --z or --u and are identical in every run. Each says how far the most extreme observed value
  sits from the centre, in that rule's units.

  Read down the column and make one judgment per variable -- is that value real?

      a value that is plausible       ->  k must be ABOVE its z
      a value that is not             ->  k must be BELOW its z

  Each variable narrows the window from one side. Any k inside the surviving window satisfies
  every judgment made, and one run is enough to find it. Rerun at that k and read the removed
  column for what it costs.

  If the window closes -- some variable's largest real value sits further out than another
  variable's worst error -- then no single threshold separates them on that scale. Under
  objective (a) that is the end of the road. Under (b) it need not be: check what the
  surviving errors actually cost. An error at u = 7 contributes 49 where a routine residual
  contributes about 1, which is a rounding error next to the values this exists to remove.
  Set --u above the largest value you believe rather than below the smallest you do not.

THE OTHER COLUMNS AND SECTIONS

  min kept / max kept   The surviving extremes under each rule. Read these before the cut
                        points, which say nothing on their own. An implausible max kept beside
                        a 'none' cut means the rule was defeated, not satisfied.

  removed / %           Counted against positive values only, since that is all either rule
                        can act on. Above {args.warn_fraction:.2%} the rule is reaching past errors and
                        into real values.

  BELOW THE LOG         Zeros and negatives. log10 is undefined there, so both rules drop them
                        before searching and neither can report on them. A small share is a
                        sentinel for a measurement not taken; a large one is the variable
                        itself. A negative reading means a cleaner ran in the wrong order.

  UNIT AUDIT            Conflicts are read from the declared UNITNAME in the variable map, so
                        they are exact and need no threshold. The medians beneath a conflict
                        only say whether the conversion actually ran.
""")


ROBUST_WIDTH = 133


def report_robust_log(by_variable, args):
    """Per variable, where a fixed number of robust spreads would cut, and how far out the
    worst surviving values actually sit. The z columns are the decision: read down them for
    a k that separates the variables you know are wrong from the ones you know are right."""
    print(f'\n{"=" * ROBUST_WIDTH}')
    print(f'ROBUST LOG SPREAD  (cut at {args.z} robust spreads from the median, in log space)')
    print('=' * ROBUST_WIDTH)
    print(f'{"variable":<34}{"spread":>10}{"z(min)":>10}{"z(max)":>10}{"low cut":>12}'
          f'{"min kept":>12}{"max kept":>13}{"high cut":>13}{"removed":>10}{"%":>9}')
    print('-' * ROBUST_WIDTH)

    for variable in sorted(by_variable):
        accumulator = by_variable[variable]
        # Median and spread come from the reservoir, which is a uniform sample. Taking them
        # from tail_sample would let the retained tails drag both.
        body = accumulator.body_sample()
        low, high, scale = robust_log_cut(body, args.z)
        positive = body[body > 0]
        centre = float(np.median(np.log10(positive))) if positive.size else np.nan
        observed_low, observed_high = accumulator.surviving(-np.inf, np.inf)
        removed, saturated = accumulator.beyond(low, high)
        smallest, largest = accumulator.surviving(low, high)
        seen = accumulator.positive_count
        share = removed / seen if seen else 0.0
        marker = '  <-- inspect' if share > args.warn_fraction else ''
        if saturated:
            marker = '  <-- SATURATED, removing far too much'
        elif not np.isfinite(scale) or scale == 0:
            marker = '  <-- no usable spread'
        print(f'{variable[:33]:<34}{_number(scale):>10}'
              f'{_number(robust_log_z(observed_low, centre, scale)):>10}'
              f'{_number(robust_log_z(observed_high, centre, scale)):>10}'
              f'{("none" if low == -np.inf else f"{low:.4g}"):>12}'
              f'{_number(smallest):>12}{_number(largest):>13}'
              f'{("none" if high == np.inf else f"{high:.4g}"):>13}'
              f'{removed:>10,}{share:>9.4%}{marker}')
    print('-' * ROBUST_WIDTH)
    print('  spread is one robust spread in decades; z(min) and z(max) are where the most '
          'extreme\n  observed values sit in those units, before any cut.')


def report_unloggable(by_variable, args):
    """Zeros and negatives, which neither log rule can see or remove.

    A zero is usually a sentinel for a measurement that was not taken, but for some
    variables it is a real reading, and the share separates the two: a handful among tens of
    thousands is a sentinel, a substantial mode is the variable.
    """
    rows = [(name, a) for name, a in sorted(by_variable.items()) if a.zeros or a.negatives]
    print(f'\n{"=" * 100}')
    print('BELOW THE LOG  (values the tail-gap and robust-spread rules cannot see)')
    print('=' * 100)
    if not rows:
        print('\n  Every observed value is positive.')
        return
    print(f'{"variable":<34}{"observed":>12}{"zeros":>10}{"%":>10}{"negatives":>12}{"%":>10}')
    print('-' * 100)
    for name, a in rows:
        print(f'{name[:33]:<34}{a.count:>12,}{a.zeros:>10,}{a.zeros / a.count:>10.3%}'
              f'{a.negatives:>12,}{a.negatives / a.count:>10.3%}')
    print('-' * 100)
    print('  A negative reading means a cleaner ran in the wrong order: remove_negative_values\n'
          '  is applied before the unit conversions that can produce one.')


STANDARDIZED_WIDTH = 145


def standardized_scale(body):
    """The centre and scale TransEHR2 standardizes with, measured robustly.

    `standardize_feats` divides by the 5th-to-95th percentile range, so that range is the
    unit a value is expressed in by the time a loss is computed on it. Unlike a MAD, it spans
    the clinically observed range rather than the regulated core, which is what a rule needs
    if real pathology is not to read as extreme.

    The centre is the median rather than the mean the model uses. A mean is dragged by the
    very values being looked for, so a diagnostic that used one would understate exactly the
    outliers it exists to find.
    """
    if body.size < 100:
        return np.nan, np.nan
    low, high = np.percentile(body, [5, 95])
    scale = float(high - low)
    return float(np.median(body)), (scale if scale > 0 else np.nan)


def standardized_extent(value, centre, scale):
    """How large a value is once standardized: the magnitude the loss actually sees."""
    if not np.isfinite(value) or not np.isfinite(scale):
        return np.nan
    return (value - centre) / scale


def report_standardized(by_variable, args):
    """Per variable, the observed extremes in the units the model standardizes to.

    This is the only one of the three rules aimed at the harm rather than at clinical
    validity: an implausible value that standardizes to single digits costs a loss nothing,
    while one that standardizes to thousands dominates every step it appears in.
    """
    print(f'\n{"=" * STANDARDIZED_WIDTH}')
    print(f'STANDARDIZED MAGNITUDE  (cut beyond {args.u} units of (p95 - p5) from the median, '
          f'the scale standardize_feats uses)')
    print('=' * STANDARDIZED_WIDTH)
    print(f'{"variable":<34}{"median":>10}{"p95 - p5":>12}{"u(min)":>11}{"u(max)":>11}'
          f'{"low cut":>12}{"min kept":>12}{"max kept":>13}{"high cut":>13}{"removed":>10}'
          f'{"%":>9}')
    print('-' * STANDARDIZED_WIDTH)

    for variable in sorted(by_variable):
        accumulator = by_variable[variable]
        centre, scale = standardized_scale(accumulator.body_sample())
        observed_low, observed_high = accumulator.surviving(-np.inf, np.inf)
        if not np.isfinite(scale):
            low, high = -np.inf, np.inf
        else:
            low, high = centre - args.u * scale, centre + args.u * scale
        removed, saturated = accumulator.beyond(low, high, positive_only=False)
        smallest, largest = accumulator.surviving(low, high)
        share = removed / accumulator.count if accumulator.count else 0.0
        marker = '  <-- inspect' if share > args.warn_fraction else ''
        if saturated:
            marker = '  <-- SATURATED, removing far too much'
        elif not np.isfinite(scale):
            marker = '  <-- no usable spread'
        print(f'{variable[:33]:<34}{_number(centre):>10}{_number(scale):>12}'
              f'{_number(standardized_extent(observed_low, centre, scale)):>11}'
              f'{_number(standardized_extent(observed_high, centre, scale)):>11}'
              f'{("none" if low == -np.inf else f"{low:.4g}"):>12}'
              f'{_number(smallest):>12}{_number(largest):>13}'
              f'{("none" if high == np.inf else f"{high:.4g}"):>13}'
              f'{removed:>10,}{share:>9.4%}{marker}')
    print('-' * STANDARDIZED_WIDTH)
    print('  u(min) and u(max) are the observed extremes in standardized units, before any\n'
          '  cut. A value at u costs the loss u^2, so these say what each variable is\n'
          '  contributing rather than whether it is clinically possible.')


UNDECLARED = '(not declared)'


def declared_unit(value):
    """One itemid's unit, normalised for comparison."""
    text = '' if value is None else str(value).strip()
    if not text or text.lower() in ('nan', 'none'):
        return UNDECLARED
    return text.casefold()


def report_unit_audit(by_itemid, var_map, args):
    """Per variable, whether its itemids declare more than one unit in the variable map.

    An itemid recorded in a unit the cleaners do not convert moves its entire distribution,
    which no per-value rule can see. Comparing itemid medians does not find it: a median
    ratio is produced just as readily by a different collection interval or a sicker
    subpopulation, and those are the majority of what it flags. The map already declares the
    unit, so the conflict is read from there, exactly and without a threshold.

    Medians still appear, but only inside a group already known to declare mixed units, and
    only to answer a second question the map cannot: whether the conversion actually ran.
    Values reaching here have been through `clean_events`, so agreeing medians mean the
    conversion is in place and diverging medians mean it is missing.
    """
    print(f'\n{"=" * 100}')
    print('UNIT AUDIT  (declared UNITNAME per itemid, from the variable map)')
    print('=' * 100)

    observed = {itemid: accumulator for (_, itemid), accumulator in by_itemid.items()}
    rows = var_map.reset_index()
    conflicts, undeclared = 0, []

    for variable, group in rows.groupby('VARIABLE'):
        entries = []
        for row in group.itertuples(index=False):
            accumulator = observed.get(row.ITEMID)
            entries.append((
                declared_unit(getattr(row, 'UNITNAME', None)),
                row.ITEMID,
                str(getattr(row, 'LABEL', '')),
                accumulator.count if accumulator is not None else 0,
                float(np.median(accumulator.body_sample())) if accumulator is not None
                else np.nan,
            ))

        units = {unit for unit, *_ in entries}
        real = units - {UNDECLARED}
        # Collected whatever the rest of the variable looks like, so the list of map
        # omissions is complete rather than only covering variables with no conflict.
        if real and UNDECLARED in units:
            undeclared.append((variable, [e for e in entries if e[0] == UNDECLARED]))
        if len(real) < 2:
            continue

        conflicts += 1
        print(f'\n  {variable}   {len(real)} declared units')
        for unit, itemid, label, count, median in sorted(entries):
            print(f'    {unit:<10}{itemid:<9}{label[:30]:<32}{count:>10,} values'
                  f'   median {_number(median):>10}')

        seen = [(unit, median) for unit, _, _, count, median in entries
                if count and np.isfinite(median) and unit != UNDECLARED]
        if len({unit for unit, _ in seen}) < 2:
            print('    -> only one of these units was observed; nothing to compare')
            continue
        medians = [median for _, median in seen]
        spread = max(medians) / min(medians) if min(medians) > 0 else np.inf
        if spread <= args.warn_ratio:
            print(f'    -> post-cleaning medians agree within {spread:.2f}x, '
                  f'so the conversion is in place')
        else:
            print(f'    -> post-cleaning medians still differ by {spread:.2f}x, '
                  f'so the conversion is MISSING')

    if not conflicts:
        print('\n  No variable pools itemids that declare different units.')

    if undeclared:
        print(f'\n  {len(undeclared)} variable(s) declare a unit for some itemids and none '
              f'for others.\n  Not a conflict on its own -- an omission in the map, which '
              f'hides one if a unit differs:')
        for variable, entries in undeclared:
            names = ', '.join(f'{itemid} ({label[:28]})' for _, itemid, label, _, _ in entries)
            print(f'    {variable}: {names}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('subjects_root', help='Directory of per-subject folders from step 1.')
    parser.add_argument('--variable_map_file', required=True)
    parser.add_argument('--subjects', type=int, default=2000,
                        help='Subjects to sample. 0 reads all of them.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gap', type=float, default=0.5,
                        help='Decades of separation that count as a detached tail. 0.5 is a '
                             'factor of about three.')
    parser.add_argument('--quantile', type=float, default=0.999)
    parser.add_argument('--min_fold', type=float, default=3.0)
    parser.add_argument('--z', type=float, default=8.0,
                        help='Robust spreads from the median, in log space, beyond which '
                             'the second rule cuts.')
    parser.add_argument('--u', type=float, default=20.0,
                        help='Units of (p95 - p5) from the median beyond which the third '
                             'rule cuts.')
    parser.add_argument('--emit_ranges', type=str, default=None,
                        help='Write the standardized-magnitude cuts to this CSV, for '
                             'extract_episodes_from_subjects.py to apply.')
    parser.add_argument('--warn_fraction', type=float, default=0.001,
                        help='Flag a variable whose cut removes more than this share.')
    parser.add_argument('--warn_ratio', type=float, default=1.5,
                        help='Within a variable that declares mixed units, how far the '
                             'post-conversion medians may differ before the conversion is '
                             'called missing.')
    args = parser.parse_args()

    if not os.path.isdir(args.subjects_root):
        raise SystemExit(f'{args.subjects_root} is not a directory')

    print_reading_guide(args)

    var_map = read_itemid_to_variable_map(args.variable_map_file)
    by_variable, by_itemid = collect(args.subjects_root, var_map, args.subjects, args.seed)
    if not by_variable:
        raise SystemExit('no events were read; check the subjects root and the variable map')

    report_tail_gap(by_variable, args)
    report_robust_log(by_variable, args)
    report_standardized(by_variable, args)
    report_unloggable(by_variable, args)
    report_unit_audit(by_itemid, var_map, args)

    if args.emit_ranges:
        emit_ranges(by_variable, args, args.emit_ranges)


if __name__ == '__main__':
    main()
