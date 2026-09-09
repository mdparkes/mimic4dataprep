"""Probes for the order of the per-value cleaners and for the outlier cuts.

The order matters in one specific way: a unit conversion with an offset can turn a
non-negative reading negative, so a negative filter placed ahead of it passes the value and
never sees the result. A temperature recorded as 0 F is the case that made this visible.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimic4dataprep.cleaners import (
    clean_events, read_variable_ranges, remove_extreme_values)
from mimic4dataprep.preprocessing import read_itemid_to_variable_map


MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'mimic4dataprep', 'resources', 'itemid_to_variable_map_used.csv')


@pytest.fixture(scope='module')
def var_map():
    return read_itemid_to_variable_map(MAP)


def cleaned(var_map, itemid, variable, value, ranges=None):
    events = pd.DataFrame({'ITEMID': [itemid], 'VARIABLE': [variable], 'VALUE': [value]})
    out = clean_events(events, var_map, ranges)
    if out.shape[0] == 0:
        return None
    return pd.to_numeric(out['VALUE'], errors='coerce').iloc[0]


def test_a_zero_fahrenheit_reading_does_not_survive_as_a_negative_celsius(var_map):
    assert pd.isna(cleaned(var_map, 223761, 'Temperature', '0'))


@pytest.mark.parametrize('reading', ['0', '10', '31.9'])
def test_no_fahrenheit_reading_below_freezing_survives_conversion(var_map, reading):
    assert pd.isna(cleaned(var_map, 223761, 'Temperature', reading))


def test_the_conversions_themselves_still_run(var_map):
    assert cleaned(var_map, 223761, 'Temperature', '98.6') == pytest.approx(37.0)
    assert cleaned(var_map, 226531, 'Weight', '176') == pytest.approx(79.83, abs=0.01)
    assert cleaned(var_map, 226707, 'Height', '68') == pytest.approx(172.72, abs=0.01)


def test_a_negative_raw_reading_is_still_removed(var_map):
    assert pd.isna(cleaned(var_map, 220045, 'Heart rate', '-5'))


def test_an_ordinary_reading_is_untouched(var_map):
    assert cleaned(var_map, 220045, 'Heart rate', '84') == pytest.approx(84.0)


RANGES = pd.DataFrame({'LOW': [-112.7, -22979.0], 'HIGH': [116.7, 23057.0]},
                      index=pd.Index(['Serum magnesium', 'AST'], name='VARIABLE'))


def frame(rows):
    return pd.DataFrame(rows, columns=['VARIABLE', 'VALUE'])


def test_a_sentinel_beyond_the_cut_is_dropped():
    kept = remove_extreme_values(frame([('AST', 21300.0), ('AST', 999999.0)]), RANGES)
    assert list(kept['VALUE']) == [21300.0]


def test_a_variable_absent_from_the_ranges_is_left_alone():
    kept = remove_extreme_values(frame([('Urine output', 999999.0)]), RANGES)
    assert kept.shape[0] == 1


def test_a_value_that_cannot_be_judged_is_left_in_place():
    kept = remove_extreme_values(frame([('AST', 'unmeasurable'), ('AST', np.nan)]), RANGES)
    assert kept.shape[0] == 2


def test_no_ranges_means_no_filtering():
    events = frame([('AST', 999999.0)])
    assert remove_extreme_values(events, None).shape[0] == 1
    assert remove_extreme_values(events, RANGES.iloc[:0]).shape[0] == 1


def test_a_ranges_file_without_the_required_columns_is_refused(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('VARIABLE,LEVEL2,OUTLIER LOW\nAST,AST,0\n')
    with pytest.raises(ValueError, match='HIGH'):
        read_variable_ranges(str(path))
