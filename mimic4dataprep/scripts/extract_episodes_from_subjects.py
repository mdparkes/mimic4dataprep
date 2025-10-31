from __future__ import absolute_import
from __future__ import print_function

import argparse
import logging
import os
from tqdm import tqdm

from mimic4dataprep.cleaners import clean_events
from mimic4dataprep.subject import read_diagnoses, read_events, read_stays
from mimic4dataprep.subject import add_hours_elapsed_to_events, get_events_for_stay
from mimic4dataprep.subject import convert_events_to_timeseries, get_last_valid_from_timeseries
from mimic4dataprep.preprocessing import assemble_episodic_data, read_itemid_to_variable_map, map_itemids_to_variables


parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--use_full_record_history', action='store_true', \
                    help="If set, episode timeseries will include all events recorded prior to and during the ICU"
                    " stay. If not set, the episode timeseries will only include "
                    "events recorded during the ICU stay.")
args, _ = parser.parse_known_args()

var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.VARIABLE.unique()

for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    subj_path = os.path.join(args.subjects_root_path, subject_dir)
    if not os.path.isdir(subj_path):
        continue
    # Get the file paths for stays, diagnoses, and events
    stays_fp = os.path.join(subj_path, 'stays.csv')
    diagnoses_fp = os.path.join(subj_path, 'diagnoses.csv')
    events_fp = os.path.join(args.subjects_root_path, subject_dir, 'events.csv')
    # Ensure the existence of the required files
    stays_exists = os.path.exists(stays_fp)
    diagnoses_exists = os.path.exists(diagnoses_fp)
    events_exists = os.path.exists(events_fp)
    # Notify if any of the required files are missing. This happens if there were no data to write to files.
    if not stays_exists:
        print(f"stays.csv file not found for subject {subject_dir}. Skipping patient.")
        continue
    if not diagnoses_exists:
        print(f"diagnoses.csv file not found for subject {subject_dir}. Skipping patient.")
        continue
    if not events_exists:
        print(f"events.csv file not found for subject {subject_dir}. Skipping patient.")
        continue
    # Reading all required tables for this subject
    stays = read_stays(subj_path)
    diagnoses = read_diagnoses(subj_path)
    try:
        events = read_events(subj_path)
    except:
        print(subj_path)
    # Filtering events to only include those that are present in the variable map
    events = map_itemids_to_variables(events, var_map)
    events = clean_events(events, var_map)
    if events.shape[0] == 0:
        print(f"Patient {subject_dir} has no events of the selected type(s). Skipping subject.")
        continue
    # Create timeseries pivot table with times in the rows and variables in the columns
    timeseries = convert_events_to_timeseries(events, variables=variables)
    # Combine the diagnoses and stay data for all ICU episodes
    episodic_data = assemble_episodic_data(stays, diagnoses)

    # extracting separate episodes
    for i in range(stays.shape[0]):

        stay_id = stays.ICUSTAY_ID.iloc[i]
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]

        # Sometimes an episode doesn't have any diagnoses, in which case we skip it.
        if stay_id not in diagnoses.ICUSTAY_ID.values:
            print(f"Patient {subject_dir}, ICU episode {i+1} has no recorded diagnoses. Skipping episode.")
            continue

        episode = get_events_for_stay(timeseries, intime, outtime, args.use_full_record_history)
        if episode.shape[0] == 0:
            print(f"Patient {subject_dir}, ICU episode {i+1} has no events of the selected type(s). Skipping episode.")
            continue

        # Timestamp events by hours elapsed relative to the start of the selected ICU stay
        episode = add_hours_elapsed_to_events(episode, intime, remove_charttime=True)
        episode = episode.set_index('HOURS').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'Weight'] = get_last_valid_from_timeseries(episode, 'Weight')
            episodic_data.loc[stay_id, 'Height'] = get_last_valid_from_timeseries(episode, 'Height')
        
        episode_path = os.path.join(subj_path, f"episode{i+1}.csv")
        episodic_data.loc[episodic_data.index == stay_id].to_csv(episode_path, index_label='Icustay')
        
        columns_sorted = sorted(list(episode.columns), key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        
        timeseries_path = os.path.join(subj_path, f"episode{i+1}_timeseries.csv")
        episode.to_csv(timeseries_path, index_label='Hours')
