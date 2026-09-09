"""Probes for the exclusion of an ICU stay's own discharge-time text.

A discharge summary and the diagnosis list that goes with it are written when an admission
ends. Extraction keeps every record charted up to ICU discharge, so without a guard a stay
contributes its own discharge documentation as a feature of itself: the diagnoses are what the
phenotype labels are derived from, and the summary recounts the outcome.

The failure is quiet in both directions, which is why each is pinned here. Too little: a
feature added to the text set and not to the guard leaks, which is how Discharge Summary came
to be unguarded while Diagnosis Descriptions was not. Too much: nullifying a record charted
before this stay discards a previous admission's summary, which is history the model is meant
to read.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimic4dataprep.subject import DISCHARGE_TIME_TEXT_FEATURES, get_events_for_stay


INTIME = pd.Timestamp('2150-06-10 12:00:00')
OUTTIME = pd.Timestamp('2150-06-14 08:00:00')


def events(offsets_hours, features=DISCHARGE_TIME_TEXT_FEATURES):
    """One row per offset from INTIME, every text feature and a vital filled in on each.

    Args:
        offsets_hours: Hours from INTIME for each row; negative is before admission.
        features: Text feature columns to include.

    Returns:
        A frame shaped like `convert_events_to_timeseries` output, with each text cell holding
        its own offset so a surviving value can be traced back to its row.
    """
    times = [INTIME + pd.Timedelta(hours=offset) for offset in offsets_hours]
    frame = pd.DataFrame({'CHARTTIME': times})
    for feature in features:
        frame[feature] = [f'{feature} at {offset:+g} h' for offset in offsets_hours]
    frame['Heart rate'] = [80.0 + offset for offset in offsets_hours]
    return frame


def surviving(result, feature):
    """Offsets whose value for `feature` was not nullified."""
    kept = result[result[feature].astype(bool)]
    return sorted(
        round((time - INTIME) / pd.Timedelta(hours=1), 6) for time in kept['CHARTTIME'])


@pytest.mark.parametrize('feature', DISCHARGE_TIME_TEXT_FEATURES)
def test_text_charted_after_admission_is_nullified(feature):
    result = get_events_for_stay(events([-72.0, -1.0, 2.0, 20.0]), INTIME, OUTTIME,
                                 use_full_history=True)
    assert surviving(result, feature) == [-72.0, -1.0]


@pytest.mark.parametrize('feature', DISCHARGE_TIME_TEXT_FEATURES)
def test_text_charted_exactly_at_admission_is_nullified(feature):
    """HOURS is measured from INTIME, so a record at INTIME is the stay's first record."""
    result = get_events_for_stay(events([-5.0, 0.0]), INTIME, OUTTIME, use_full_history=True)
    assert surviving(result, feature) == [-5.0]


@pytest.mark.parametrize('feature', DISCHARGE_TIME_TEXT_FEATURES)
def test_text_from_a_previous_admission_is_kept(feature):
    """The guard must not reach backwards; this is the history the features exist to carry."""
    result = get_events_for_stay(events([-8760.0, -730.0, -0.5]), INTIME, OUTTIME,
                                 use_full_history=True)
    assert surviving(result, feature) == [-8760.0, -730.0, -0.5]


def test_a_non_text_feature_is_left_alone():
    """Only the discharge-time text is dropped; in-stay vitals are the point of the record."""
    frame = events([-2.0, 0.0, 6.0])
    result = get_events_for_stay(frame, INTIME, OUTTIME, use_full_history=True)
    assert result['Heart rate'].tolist() == [78.0, 80.0, 86.0]


@pytest.mark.parametrize('feature', DISCHARGE_TIME_TEXT_FEATURES)
def test_the_guard_applies_without_full_history(feature):
    """The stay-only mode keeps [INTIME, OUTTIME], so all of its text is the stay's own."""
    result = get_events_for_stay(events([-3.0, 1.0, 30.0]), INTIME, OUTTIME,
                                 use_full_history=False)
    assert surviving(result, feature) == []


def test_a_missing_text_column_is_not_an_error():
    """Not every extraction requests every text feature."""
    frame = events([-4.0, 5.0], features=('Discharge Summary',))
    result = get_events_for_stay(frame, INTIME, OUTTIME, use_full_history=True)
    assert surviving(result, 'Discharge Summary') == [-4.0]
    assert 'Diagnosis Descriptions' not in result.columns


def test_every_declared_text_feature_is_covered():
    """The guard iterates the declaration, so the two cannot fall out of step -- which is the
    defect that left Discharge Summary unguarded."""
    frame = events([1.0])
    result = get_events_for_stay(frame, INTIME, OUTTIME, use_full_history=True)
    for feature in DISCHARGE_TIME_TEXT_FEATURES:
        assert surviving(result, feature) == [], feature
