from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import pandas as pd

from mimic4dataprep.util import dataframe_from_csv


def read_stays(subject_path):
    stays = dataframe_from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays


def read_diagnoses(subject_path):
    return dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)


def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events[events.VALUE.notnull()]


    # Try standard datetime conversion first
    try:
        events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    except ValueError:
        # If it fails, try handling dates without times
        events.CHARTTIME = events.CHARTTIME.astype(str)
        # Find entries that are just dates (10 characters) and append midnight time
        mask = events.CHARTTIME.str.len() == 10
        events.loc[mask, 'CHARTTIME'] = events.loc[mask, 'CHARTTIME'] + ' 00:00:00'
        # Convert to datetime
        events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
        
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    # events.sort_values(by=['CHARTTIME', 'ITEMID', 'ICUSTAY_ID'], inplace=True)
    return events


def get_events_for_stay(events, intime, outtime, use_full_history=False):
    
    if use_full_history:
        # Excludes events with a CHARTTIME that occurs after the ICU stay OUTTIME
        idx = events.CHARTTIME <= outtime
    elif intime is not None and outtime is not None:
        # Only include events that occur within the ICU stay

        # Original code (just to give context to the revision)
        #
        # Because of the steps performed in validate_events.py, ICUSTAY_ID is assigned to events from the
        # same hospital admission even if they occur outside the ICU stay. Consequently, the `|` operator below 
        # supersedes the CHARTTIME logic if validate_events.py was run without --use_full_history. In other words,
        # the code below ignores CHARTTIME completely if the ICU stay ID matches, even if the CHARTTIME is outside the
        # ICU stay.
        #
        # idx = (events.ICUSTAY_ID == icustayid)
        # if intime is not None and outtime is not None:
        #     idx = idx | ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))

        idx = ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))
    else:
        raise ValueError('intime and outtime must be supplied as arguments')

    events = events[idx]

    if 'ICUSTAY_ID' in events.columns:
        del events['ICUSTAY_ID']

    # Drop ICD diagnoses features that were recorded during the current hospital stay.
    # The current stay's diagnoses determine phenotype labels and cannot be used as features.
    # Locate non-null diagnosis description values and nullify them if CHARTTIME > intime
    if 'Diagnosis Descriptions' in events.columns:
        sel = events['Diagnosis Descriptions'].notnull() & (events.CHARTTIME > intime)
        if sel.any():
            events.loc[sel, 'Diagnosis Descriptions'] = ''

    return events


def add_hours_elapsed_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events['HOURS'] = (events.CHARTTIME - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del events['CHARTTIME']
    return events


def convert_events_to_timeseries(events, variable_column='VARIABLE', variables=[]):

    timeseries = events[['CHARTTIME', variable_column, 'VALUE']]
    timeseries = timeseries.sort_values(by=['CHARTTIME', variable_column, 'VALUE'], axis=0)
    timeseries = timeseries.drop_duplicates(subset=['CHARTTIME', variable_column], keep='last')
    timeseries = timeseries.pivot(index='CHARTTIME', columns=variable_column, values='VALUE')
    timeseries = timeseries.sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries

# This version of the function is a modification of the original. The original function selected the first valid value
# of the selected variable from the timeseries (events table). When the only events used are from the hospital admission
# where the ICU stay occurred, this is probably fine if we don't expect the variable to vary much. But if the entire
# history of records is used, it would be better to select the last valid value before the ICU stay. This is especially
# important if the variable is expected to change throughout the patient's history. Therefore, this version of the
# function selects the last valid value that was recorded before the ICU stay.
def get_last_valid_from_timeseries(timeseries, variable):
    # If the index is not 'HOURS', raise an exception
    if timeseries.index.name != 'HOURS':
        raise ValueError('The index of the timeseries must be "HOURS", and should express time elapsed relative to the '
                         'start of the ICU stay.')
    if variable in timeseries:
        idx = (timeseries[variable].notnull()) & (timeseries.index < 0.)
        if idx.any():
            loc = np.where(idx)[0][-1]
            return timeseries[variable].iloc[loc]
    return np.nan
