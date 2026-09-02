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
                runs.

    unit audit  Compare each ITEMID's median against its variable's. A whole ITEMID recorded in
                the wrong unit is invisible per value -- every reading is a plausible number --
                but it moves the entire distribution, so it shows up as a median ratio.

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

    def beyond(self, low, high):
        """Exactly how many observed values fall outside the cuts.

        Counted from the retained extremes rather than the reservoir, so it is exact rather
        than sampled -- unless more than EXTREMES values lie beyond a cut, which would mean
        the rule is removing far too much to adopt anyway.
        """
        tail_high, tail_low = self.high, self.low
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

    by_variable, by_itemid, failures = {}, {}, 0
    for index, name in enumerate(directories, 1):
        try:
            events = read_subject(os.path.join(subjects_root, name), var_map)
        except Exception:
            failures += 1
            continue
        if events is None:
            continue
        numeric = events[['VARIABLE', 'ITEMID', 'VALUE']].copy()
        numeric['VALUE'] = numeric['VALUE'].astype(float, errors='ignore')
        for variable, group in numeric.groupby('VARIABLE'):
            values = group['VALUE'].to_numpy(dtype=float, na_value=np.nan)
            values = values[np.isfinite(values)]
            if values.size:
                by_variable.setdefault(variable, Accumulator()).add(values)
            for itemid, sub in group.groupby('ITEMID'):
                v = sub['VALUE'].to_numpy(dtype=float, na_value=np.nan)
                v = v[np.isfinite(v)]
                if v.size:
                    by_itemid.setdefault(
                        (variable, itemid),
                        Accumulator(ITEMID_RESERVOIR, ITEMID_EXTREMES)).add(v)
        if index % 500 == 0:
            print(f'  {index:,} subjects, {len(by_variable)} variables', flush=True)

    if failures:
        print(f'  {failures:,} subjects could not be read and were skipped')
    return by_variable, by_itemid


def report_tail_gap(by_variable, args):
    """Per variable, where the rule would cut and how much it would remove."""
    print(f'\n{"=" * 100}')
    print(f'TAIL GAP  (gap > {args.gap} decades, beyond q{args.quantile}, '
          f'no closer than {args.min_fold}x the median)')
    print('=' * 100)
    print(f'{"variable":<34}{"observed":>11}{"median":>11}{"low cut":>12}{"high cut":>14}'
          f'{"removed":>10}{"%":>8}')
    print('-' * 100)

    for variable in sorted(by_variable):
        accumulator = by_variable[variable]
        low, high = tail_gap_cut(accumulator.tail_sample(),
                                 args.gap, args.quantile, args.min_fold)
        removed, saturated = accumulator.beyond(low, high)
        share = removed / accumulator.count if accumulator.count else 0.0
        marker = '  <-- inspect' if share > args.warn_fraction else ''
        if saturated:
            marker = '  <-- SATURATED, removing far too much'
        print(f'{variable[:33]:<34}{accumulator.count:>11,}'
              f'{np.median(accumulator.body_sample()):>11.3g}'
              f'{("none" if low == -np.inf else f"{low:.4g}"):>12}'
              f'{("none" if high == np.inf else f"{high:.4g}"):>14}'
              f'{removed:>10,}{share:>8.4%}{marker}')
    print('-' * 100)
    print(f'  Counts are exact while a tail holds fewer than its cap of retained values. '
          f'in full.')
    print(f'  Anything above {args.warn_fraction:.2%} is removing more than errors and '
          f'wants looking at before the rule is adopted.')


def report_unit_audit(by_variable, by_itemid, args):
    """Per itemid, the median ratio against its variable. A whole itemid in the wrong unit
    moves its entire distribution, which no per-value rule can see."""
    print(f'\n{"=" * 100}')
    print(f'UNIT AUDIT  (itemid median vs variable median, flagged beyond {args.warn_ratio}x)')
    print('=' * 100)

    flagged = 0
    for variable in sorted(by_variable):
        variable_median = float(np.median(by_variable[variable].body_sample()))
        if not np.isfinite(variable_median) or variable_median == 0:
            continue
        entries = [(itemid, accumulator) for (name, itemid), accumulator in by_itemid.items()
                   if name == variable]
        if len(entries) < 2:
            continue
        rows = []
        for itemid, accumulator in entries:
            median = float(np.median(accumulator.body_sample()))
            ratio = median / variable_median
            if not (1 / args.warn_ratio < ratio < args.warn_ratio):
                rows.append((itemid, accumulator.count, median, ratio))
        if rows:
            flagged += len(rows)
            print(f'\n  {variable}   variable median {variable_median:.4g}')
            for itemid, count, median, ratio in sorted(rows, key=lambda r: -abs(r[3])):
                print(f'    itemid {itemid:<10}{count:>10,} values   median {median:>12.4g}'
                      f'   {ratio:>8.2f}x')
    if not flagged:
        print('\n  No itemid median departs from its variable by more than '
              f'{args.warn_ratio}x.')
    else:
        print(f'\n  {flagged} itemid(s) flagged. A consistent multiple is a unit the cleaners '
              f'did not convert.\n  The variable median sits in whichever mode has more '
              f'values, so the flagged itemid is the\n  minority one and not necessarily the '
              f'wrong one -- check both against UNITNAME in the\n  variable map before '
              f'changing anything.')


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
    parser.add_argument('--warn_fraction', type=float, default=0.001,
                        help='Flag a variable whose cut removes more than this share.')
    parser.add_argument('--warn_ratio', type=float, default=1.5)
    args = parser.parse_args()

    if not os.path.isdir(args.subjects_root):
        raise SystemExit(f'{args.subjects_root} is not a directory')

    var_map = read_itemid_to_variable_map(args.variable_map_file)
    by_variable, by_itemid = collect(args.subjects_root, var_map, args.subjects, args.seed)
    if not by_variable:
        raise SystemExit('no events were read; check the subjects root and the variable map')

    report_tail_gap(by_variable, args)
    report_unit_audit(by_variable, by_itemid, args)


if __name__ == '__main__':
    main()
