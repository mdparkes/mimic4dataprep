"""
This script performs filtering on the events.csv file based on the contents of the stays.csv file. The script only works
with stays.csv files that were generated to include only one ICU stay per hospital admission. The script applies the
following steps:

- Remove all entries in events.csv where HADM_ID (hospital admission ID) is empty.
- Remove all entries in events.csv where HADM_ID is not listed in stays.csv.
- Fills in empty ICUSTAY_ID (ICU stay ID) in events.csv using values from stays.csv. Note that if an event was recorded 
  during a hospital admission which included an ICU stay, but the event was not recorded during the ICU stay, the event 
  still gets labeled with the ICUSTAY_ID from stays.csv.
- Removes admissions from events.csv where missing ICUSTAY_ID could not be recovered from stays.csv.
- Removes events where the ICUSTAY_ID in events.csv does not match the ICUSTAY_ID in stays.csv for the same HADM_ID. The
  only reason this should happen is if the ICUSTAY_ID was mistyped somewhere, as earlier filtering steps ensured that
  the remaining events are all associated with hospital admissions in stays.csv, implying that each remaining hospital
  admission in events.csv has only one ICU stay associated with it.

If performing experiments that require a complete history of events from all prior hospital admissions for each ICU
stay, the '--use_full_record_history' flag should be set in the command line args. In this case only the following
steps will be taken:

- Remove all entries in events.csv where HADM_ID (hospital admission ID) is empty.
- Given HADM_ID, if an event CHARTTIME is between the ICU INTIME and OUTTIME in stays and ICUSTAY_ID is not recorded for
  that event in the event table, fill it in with the corresponding ICUSTAY_ID from the stays table.
- Given a non-missing ICUSTAY_ID from events and a non-missing ICUSTAY_ID from stays, remove the event if the IDs are
  not the same.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
from tqdm import tqdm


def is_subject_folder(x):
    return str.isdigit(x)


def load_table(table, root_path):
    table_path = os.path.join(root_path, table + '.csv')
    if not os.path.exists(table_path):
        raise ValueError('Table {} does not exist.'.format(table))
    df = pd.read_csv(table_path, index_col=False, dtype={'HADM_ID': str, "ICUSTAY_ID": str})
    df.columns = df.columns.str.upper()
    if table == 'stays':
        assert(not df['ICUSTAY_ID'].isnull().any())
        assert(not df['HADM_ID'].isnull().any())
        assert(len(df['ICUSTAY_ID'].unique()) == len(df['ICUSTAY_ID']))
    return df


def main():

    n_events = 0                   # total number of events
    empty_hadm = 0                 # HADM_ID is empty in events.csv. We exclude such events.
    no_hadm_in_stay = 0            # HADM_ID does not appear in stays.csv. We exclude such events.
    no_icustay = 0                 # ICUSTAY_ID is empty in events.csv. We try to fix such events.
    recovered = 0                  # empty ICUSTAY_IDs are recovered according to stays.csv files (given HADM_ID)
    could_not_recover = 0          # empty ICUSTAY_IDs that are not recovered. This should be zero.
    icustay_missing_in_stays = 0   # ICUSTAY_IDs in events that do not appear in stays but w/ same HADM_ID. Excluded.

    parser = argparse.ArgumentParser()
    parser.add_argument('subjects_root_path', type=str,
                        help='Directory containing subject subdirectories.')
    parser.add_argument('--use_full_record_history', action='store_true')
    args = parser.parse_args()
    print(args)

    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))


    if args.use_full_record_history:
        # Filtering steps for the case where the complete history of events from all prior hospital admissions is needed

        for subject in tqdm(subjects, desc='Iterating over subjects'):
            stays_df = load_table('stays', os.path.join(args.subjects_root_path, subject))
            events_df = load_table('events', os.path.join(args.subjects_root_path, subject))
            n_events += events_df.shape[0]

            # Drop all events where HADM_ID was not recorded
            empty_hadm += events_df['HADM_ID'].isnull().sum()
            events_df = events_df.dropna(subset=['HADM_ID'])

            merged_df = events_df.merge(
                stays_df, left_on=['HADM_ID'], right_on=['HADM_ID'], how='left', suffixes=['', '_r'], indicator=True
            )
            # Number of events associated with admissions that are not in stays.csv
            no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
            # Select events with HADM_IDs that appear in stays.csv but have no recorded ICUSTAY_ID
            sel_both = (merged_df['_merge'] == 'both') & (merged_df['ICUSTAY_ID'].isnull())
            # If CHARTTIME is between INTIME and OUTTIME (inclusive), fill 'ICUSTAY_ID' with 'ICUSTAY_ID_r' from stays
            sel_time_lb = merged_df['CHARTTIME'] >= merged_df['INTIME']
            sel_time_ub = merged_df['CHARTTIME'] <= merged_df['OUTTIME']
            sel_time = sel_time_lb & sel_time_ub
            sel = sel_both & sel_time
            recovered += sel.sum()  # Number of ICUSTAY_ID values inferred from stays table
            merged_df.loc[sel, 'ICUSTAY_ID'] = merged_df.loc[sel, 'ICUSTAY_ID_r']
            
            # Remove events that have different non-null ICUSTAY_ID and ICUSTAY_ID_r values
            sel_non_null = (~merged_df['ICUSTAY_ID'].isnull()) & (~merged_df['ICUSTAY_ID_r'].isnull())
            sel_mismatched = merged_df['ICUSTAY_ID'] != merged_df['ICUSTAY_ID_r']
            sel = sel_non_null & sel_mismatched
            icustay_missing_in_stays += sel.sum()
            merged_df.drop(index=merged_df[sel].index, inplace=True)

            # print(f'Original number of events: {n_events}')
            # print(f'Removed {empty_hadm} events with no recorded HADM_ID')
            # print(f'Identified {no_hadm_in_stay} events from hospital stays that were not present in stays.csv')
            # print(f'Inferred ICUSTAY_IDs for {recovered} events using information from stays.csv')
            # print(f'Removed {icustay_missing_in_stays} events with inconsistent ICUSTAY_ID in events.csv and stays.csv')
            # print('\n')

            to_write = merged_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']]
            to_write.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)

    else:

        for subject in tqdm(subjects, desc='Iterating over subjects'):
            stays_df = load_table('stays', os.path.join(args.subjects_root_path, subject))
            events_df = load_table('events', os.path.join(args.subjects_root_path, subject))
            n_events += events_df.shape[0]

            # Assert there are no repetitions of HADM_ID since admissions. This means that this script is only
            # compatible with stays.csv files that were generated to include only one ICU stay per hospital admission.
            assert(len(stays_df['HADM_ID'].unique()) == len(stays_df['HADM_ID']))

            # we drop all events for them HADM_ID is empty
            # TODO: maybe we can recover HADM_ID by looking at ICUSTAY_ID
            empty_hadm += events_df['HADM_ID'].isnull().sum()
            events_df = events_df.dropna(subset=['HADM_ID'])

            # ICU stays without an HADM_ID that maps to events.csv (that is, ICU stays with no recorded events) are
            # not included in the merged dataframe, but the left merge ensures that events from admissions not present 
            # in stays.csv are still included in the merged dataframe (at least until the next step).
            merged_df = events_df.merge(stays_df, left_on=['HADM_ID'], right_on=['HADM_ID'],
                                        how='left', suffixes=['', '_r'], indicator=True)

            # no_hadm_in_stay is number of events associated with admissions that are not in stays.csv
            no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
            # We drop all events for which HADM_ID is not listed in stays.csv since there is no way to know the targets 
            # of that stay (for example mortality). In other words, we only keep events from hospital admissions that 
            # have an ICU stay recorded in stays.csv.
            merged_df = merged_df[merged_df['_merge'] == 'both']

            # if ICUSTAY_ID is empty in stays.csv, we try to recover it
            # we exclude all events for which we could not recover ICUSTAY_ID
            # This assumes that events from admissions not in stays.csv have already been dropped, and only works 
            # because earlier filtering steps in the creation of stays.csv ensure that each HADM_ID in stays.csv maps 
            # to a single ICUSTAY_ID.
            cur_no_icustay = merged_df['ICUSTAY_ID'].isnull().sum()
            no_icustay += cur_no_icustay
            # Fill in missing ICUSTAY_IDs in events using ICUSTAY_IDs from stays (given matching HADM_ID)
            # Subtlety: Even if an event from a hospital admission in stays.csv was not recorded during the ICU stays, 
            # it still gets labeled with the ICUSTAY_ID from stays.csv.
            merged_df.loc[:, 'ICUSTAY_ID'] = merged_df['ICUSTAY_ID'].fillna(merged_df['ICUSTAY_ID_r'])
            recovered += cur_no_icustay - merged_df['ICUSTAY_ID'].isnull().sum()
            could_not_recover += merged_df['ICUSTAY_ID'].isnull().sum()
            merged_df = merged_df.dropna(subset=['ICUSTAY_ID'])

            # now we take a look at the case when ICUSTAY_ID is present in events.csv, but not in stays.csv
            # this mean that ICUSTAY_ID in events.csv is not the same as that of stays.csv for the same HADM_ID
            # we drop all such events NOTE I assume this would only happen if the ICUSTAY_ID was mistyped somewhere,
            # because the stays.csv file only contains admissions with a defined range in the number of ICU stays,
            # and all admissions that did not appear in stays.csv were dropped in an earlier step.
            icustay_missing_in_stays += (merged_df['ICUSTAY_ID'] != merged_df['ICUSTAY_ID_r']).sum()
            merged_df = merged_df[(merged_df['ICUSTAY_ID'] == merged_df['ICUSTAY_ID_r'])]

            assert(could_not_recover == 0)
            print('n_events: {}'.format(n_events))
            print('empty_hadm: {}'.format(empty_hadm))
            print('no_hadm_in_stay: {}'.format(no_hadm_in_stay))
            print('no_icustay: {}'.format(no_icustay))
            print('recovered: {}'.format(recovered))
            print('could_not_recover: {}'.format(could_not_recover))
            print('icustay_missing_in_stays: {}'.format(icustay_missing_in_stays))
            print('\n')

            to_write = merged_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']]
            to_write.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)


if __name__ == "__main__":
    main()
