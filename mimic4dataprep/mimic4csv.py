from __future__ import absolute_import
from __future__ import print_function

import csv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from mimic4dataprep.util import dataframe_from_csv


def read_discharge_table(mimic4_notes_path):
    notes = dataframe_from_csv(os.path.join(mimic4_notes_path, 'note', 'discharge.csv'))
    notes = notes[['NOTE_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]
    notes.CHARTTIME = pd.to_datetime(notes.CHARTTIME)
    return notes


def read_patients_table(mimic4_path):
    pats = dataframe_from_csv(os.path.join(mimic4_path, 'hosp', 'patients.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOD','ANCHOR_AGE']]
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats


def read_admissions_table(mimic4_path):
    admits = dataframe_from_csv(os.path.join(mimic4_path, 'hosp', 'admissions.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic4_path):
    stays = dataframe_from_csv(os.path.join(mimic4_path, 'icu', 'icustays.csv'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays


def read_icd_diagnoses_table(mimic4_path):
    codes = dataframe_from_csv(os.path.join(mimic4_path, 'hosp', 'd_icd_diagnoses.csv'))
    codes = codes[['ICD_CODE', 'ICD_VERSION', 'LONG_TITLE']]
    codes['ICD_VERSION'] = codes['ICD_VERSION'].astype(int)  # Ensure ICD_VERSION is an integer
    diagnoses = dataframe_from_csv(os.path.join(mimic4_path, 'hosp', 'diagnoses_icd.csv'))
    diagnoses['ICD_VERSION'] = diagnoses['ICD_VERSION'].astype(int)  # Ensure ICD_VERSION is an integer
    diagnoses = diagnoses.merge(
        codes, how='inner', left_on=['ICD_CODE', 'ICD_VERSION'], right_on=['ICD_CODE', 'ICD_VERSION']
    )
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses


def get_table_file_path(mimic4_path, table):
    for root, dirs, files in os.walk(mimic4_path):
        for file in files:
            if table in file:
                return os.path.join(root, file)
    return None


def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[['ICD_CODE', 'ICD_VERSION', 'LONG_TITLE']].drop_duplicates()
    codes = codes.set_index(['ICD_CODE', 'ICD_VERSION'])
    codes['COUNT'] = diagnoses.groupby(['ICD_CODE', 'ICD_VERSION'])['ICUSTAY_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes[codes.COUNT > 0]
    if output_path:
        codes.to_csv(output_path, index_label=['ICD_CODE', 'ICD_VERSION'])
    return codes.sort_values('COUNT', ascending=False).reset_index()


def remove_icustays_with_transfers(stays):
    stays = stays[ (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'INTIME', 'OUTTIME', 'LOS']]


def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def add_age_to_icustays(stays,patients):
    stays['AGE'] = patients['ANCHOR_AGE']
    return stays


def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays


def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep[(to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays


def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays[(stays.AGE >= min_age) & (stays.AGE <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        if not os.path.exists(dn):
            os.makedirs(dn)
        if subject_id in stays.SUBJECT_ID.values:
            df_out = stays[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME')
            df_out.to_csv(os.path.join(dn, 'stays.csv'), index=False)


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        if not os.path.exists(dn):
            os.makedirs(dn)
        if subject_id in diagnoses.SUBJECT_ID.values:
            df_out = diagnoses[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM'])
            df_out.to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)


def count_table_rows(mimic4_path, table):
    file_path = get_table_file_path(mimic4_path, table)
    with open(file_path, 'rb') as f:
        # Count number of newlines, subtract 1 for header
        return sum(1 for _ in f) - 1


def read_events_table_by_row(mimic4_path, table, column_names, subjects_to_keep=None, items_to_keep=None, chunksize=20000):
    file_path = get_table_file_path(mimic4_path, table)
    rename_mapping = {
        'STAY_ID': 'ICUSTAY_ID',
        'CAREGIVER_ID': 'CGID',
        'RACE': 'ETHNICITY',
        'TEXT': 'VALUE'
    }
    
    # Process full chunks rather than individual rows
    for chunk in pd.read_csv(file_path, chunksize=chunksize, quoting=csv.QUOTE_MINIMAL):
        chunk.columns = chunk.columns.str.upper()
        chunk = chunk.rename(columns=rename_mapping)
        
        if table == 'discharge':
            chunk['ITEMID'] = 1000000
            
        # Fill missing columns
        for col in column_names:
            if col not in chunk.columns:
                chunk[col] = pd.NA
                
        chunk = chunk[column_names]
        
        if subjects_to_keep is not None:
            chunk = chunk[chunk['SUBJECT_ID'].isin(subjects_to_keep)]
        
        if items_to_keep is not None:
            chunk = chunk[chunk['ITEMID'].isin(items_to_keep)]
            
        yield chunk


def read_events_table_and_break_up_by_subject(
    input_dir, table, output_dir, items_to_keep=None, subjects_to_keep=None, chunksize=50000
):
    """Reads an events table from MIMIC and breaks it up by subject.
    
    Note that the progress bar will show the actual number of rows processed over the total number of rows in the table.
    """
    
    if items_to_keep is not None:
        items_to_keep = set(items_to_keep)

    if subjects_to_keep is not None:
        subjects_to_keep = set(subjects_to_keep)

    col_names = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    missing_subject_id_count = 0
    
    # For storing data by subject_id
    subject_buffers = {}
    
    # Process chunks of data rather than individual rows
    reader = read_events_table_by_row(
        input_dir, table, col_names, subjects_to_keep=subjects_to_keep, 
        items_to_keep=items_to_keep, chunksize=chunksize
    )
    n_rows = count_table_rows(input_dir, table)
    desc = f'Processing {table} table'
    
    # Create progress bar based on total rows
    pbar = tqdm(total=n_rows, desc=desc)
    rows_processed = 0
    
    for chunk in reader:
        # Update progress bar with the current chunk size
        chunk_size = len(chunk)
        rows_processed += chunk_size
        
        # Count and remove rows with missing subject IDs
        missing_count = chunk['SUBJECT_ID'].isna().sum()
        missing_subject_id_count += missing_count
        chunk = chunk.dropna(subset=['SUBJECT_ID'])
        
        # Group data by subject_id
        for subject_id, group in chunk.groupby('SUBJECT_ID'):
            if subject_id not in subject_buffers:
                subject_buffers[subject_id] = []
            subject_buffers[subject_id].append(group)
            
            # Write to disk if buffer gets too large
            if len(subject_buffers[subject_id]) > 10:
                subject_dir = os.path.join(output_dir, str(int(subject_id)))
                os.makedirs(subject_dir, exist_ok=True)
                output_file = os.path.join(subject_dir, 'events.csv')
                
                combined_data = pd.concat(subject_buffers[subject_id], ignore_index=True)
                
                # Write header if file doesn't exist
                file_exists = os.path.isfile(output_file)
                combined_data.to_csv(
                    output_file, 
                    index=False, 
                    mode='a' if file_exists else 'w',
                    header=not file_exists,
                    quoting=csv.QUOTE_MINIMAL
                )
                subject_buffers[subject_id] = []
            
        pbar.update(chunk_size)
    # Close the progress bar
    pbar.close()
    
    # Write remaining data for each subject
    for subject_id, buffers in subject_buffers.items():
        if buffers:
            subject_dir = os.path.join(output_dir, str(int(subject_id)))
            os.makedirs(subject_dir, exist_ok=True)
            output_file = os.path.join(subject_dir, 'events.csv')
            
            combined_data = pd.concat(buffers, ignore_index=True)
            
            # Write header if file doesn't exist
            file_exists = os.path.isfile(output_file)
            combined_data.to_csv(
                output_file, 
                index=False, 
                mode='a' if file_exists else 'w',
                header=not file_exists,
                quoting=csv.QUOTE_MINIMAL
            )
    
    # Print the number of rows with missing SUBJECT_ID
    if missing_subject_id_count > 0:
        print(f'Skipped {missing_subject_id_count} rows with missing SUBJECT_ID in {table} table.\n')
        