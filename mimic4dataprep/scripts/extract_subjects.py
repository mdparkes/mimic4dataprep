from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml

from mimic4dataprep.mimic4csv import *
from mimic4dataprep.preprocessing import make_phenotype_label_matrix
from mimic4dataprep.preprocessing import add_hcup_groups
from mimic4dataprep.util import dataframe_from_csv

ICD_DIAGNOSIS_ITEMID = 1000001  # ITEMID for ICD diagnosis codes in events.csv

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-IV CSV files.')
parser.add_argument('mimic4_path', type=str, help='Directory containing MIMIC-IV CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['chartevents', 'labevents', 'outputevents'])
parser.add_argument('--icd9_phenotype_definitions', '-p9', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                    help='YAML file with ICD-9 phenotype definitions.')
parser.add_argument('--icd10_phenotype_definitions', '-p10', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccsr_2024_definitions.yaml'),
                    help='YAML file with ICD-10-CM phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
parser.add_argument('--notes_path', type=str, help='Directory containing MIMIC-IV-Note CSV files.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass

patients = read_patients_table(args.mimic4_path)
admits = read_admissions_table(args.mimic4_path)
stays = read_icustays_table(args.mimic4_path)
if args.verbose:
    print('START:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}\n'.format(
        stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]
    ))

stays = remove_icustays_with_transfers(stays)
if args.verbose:
    print('REMOVE ICU TRANSFERS:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}\n'.format(
        stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]
    ))

stays = merge_on_subject_admission(stays, admits)  # Only retain admissions with ICU stays
stays = merge_on_subject(stays, patients)
stays = filter_admissions_on_nb_icustays(stays)  # Uses default: exclude admissions with multiple ICU stays
if args.verbose:
    print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}\n'.format(
        stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]
    ))

stays = add_age_to_icustays(stays, patients)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)
if args.verbose:
    print('REMOVE PATIENTS AGE < 18:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}\n'.format(
        stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]
    ))
stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)

diagnoses = read_icd_diagnoses_table(args.mimic4_path)
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))

icd9_defs = yaml.load(open(args.icd9_phenotype_definitions, 'r'), Loader=yaml.Loader)
icd10_defs = yaml.load(open(args.icd10_phenotype_definitions, 'r'), Loader=yaml.Loader)

# Adds HCUP_CCS_2015 (for ICD 9 codes) and HCUP_CCSR_2024 (for ICD 10 codes) columns to diagnoses
# Also adds a USE_IN_BENCHMARK column to diagnoses -- for the phenotype prediction benchmark task
diagnoses = add_hcup_groups(diagnoses, icd_version=9, definitions=icd9_defs)
diagnoses = add_hcup_groups(diagnoses, icd_version=10, definitions=icd10_defs)

path_out = os.path.join(args.output_path, 'phenotype_labels.csv')
plmat = make_phenotype_label_matrix(diagnoses, icd9_defs, icd10_defs, stays)
plmat.to_csv(path_out, index=False, quoting=csv.QUOTE_NONNUMERIC)

if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    # stays = stays.merge(patients[['SUBJECT_ID']], left_on='SUBJECT_ID', right_on='SUBJECT_ID')
    stays = stays[stays.SUBJECT_ID.isin(patients.SUBJECT_ID)]
    # args.event_tables = [args.event_tables[0]]
    # print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')

subjects = stays.SUBJECT_ID.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects)  # ICU stays that met criteria, stays.csv
break_up_diagnoses_by_subject(diagnoses, args.output_path, subjects=subjects)
if args.itemids_file is not None:  # None by default if not supplied
    items_to_keep = [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]
else:
    items_to_keep = None  # Default
for table in args.event_tables:
    # The result, events.csv, should have events from all admissions of each subject, not just selected ICU stays
    read_events_table_and_break_up_by_subject(
        input_dir=args.notes_path if table == 'discharge' else args.mimic4_path,
        table=table,
        output_dir=args.output_path,
        items_to_keep=items_to_keep,
        subjects_to_keep=subjects
    )

# TODO Review the following code and ensure it is correct.
# Add descriptions of ICD diagnosis codes during each hospital stay to events.csv as a pipe-delimited string
if ICD_DIAGNOSIS_ITEMID in items_to_keep:  # 1000001 is the ITEMID for the concatenated string of ICD diagnoses
    if args.verbose:
        print('Adding ICD diagnosis descriptions to events.csv')

    for subject_dir in tqdm(os.listdir(args['output_path']), desc='Iterating over subjects'):
            subj_path = os.path.join(args['output_path'], subject_dir)
            if not os.path.isdir(subj_path):
                continue

            stays_path = os.path.join(subj_path, 'stays.csv')
            if not os.path.exists(stays_path):
                continue
            stays = pd.read_csv(stays_path)

            diagnoses_path = os.path.join(subj_path, 'diagnoses.csv')
            if not os.path.exists(diagnoses_path):
                continue
            diagnoses = pd.read_csv(diagnoses_path)

            events_path = os.path.join(subj_path, 'events.csv')
            events = pd.read_csv(events_path)

            # Collect new rows for events dataframe
            new_events = []

            # Process each hospital admission
            for _, stay in stays.iterrows():
                hadm_id = stay['HADM_ID']
                subject_id = stay['SUBJECT_ID']
                dischtime = stay['DISCHTIME']
                
                # Get all diagnoses for this hospital admission
                hadm_diagnoses = diagnoses[diagnoses['HADM_ID'] == hadm_id]
                
                # Skip if no diagnoses found
                if hadm_diagnoses.empty:
                    continue
                
                # Create pipe-delimited string of LONG_TITLE values
                diagnoses_str = '|'.join(hadm_diagnoses['LONG_TITLE'].dropna())
                
                # Create new row for events dataframe
                new_event = {
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': hadm_id,
                    'ICUSTAY_ID': '',
                    'CHARTTIME': pd.to_datetime(dischtime),
                    'ITEMID': ICD_DIAGNOSIS_ITEMID,
                    'VALUE': diagnoses_str,
                    'VALUEUOM': ''
                }
                
                # Append new row to the list
                new_events.append(new_event)
            
            # Append all new rows to events.csv
            if new_events:  # Only append if there are new events to add
                # Convert new events to a DataFrame
                new_events_df = pd.DataFrame(new_events)
                # Append to the CSV file without writing header
                new_events_df.to_csv(events_path, mode='a', header=False, index=False, quoting=csv.QUOTE_MINIMAL)
