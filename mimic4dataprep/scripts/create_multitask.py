from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
import yaml
import random
import re

from datetime import datetime
from tqdm import tqdm

from mimic4dataprep.util import get_resources_dir_path

random.seed(49297)


def process_partition(root_dir, output_dir, partition, file_table, definitions, sample_rate=1.0, shortest_length=4,
                      eps=1e-6, future_time_interval=24.0, fixed_hours=48.0):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    icd9_code_to_group = {}
    for group in definitions:
        codes = definitions[group]['icd9_codes']
        for code in codes:
            if code not in icd9_code_to_group:
                icd9_code_to_group[code] = [group]
            else:
                icd9_code_to_group[code].append(group)

    icd10_code_to_group = {}
    for group in definitions:
        codes = definitions[group]['icd10_codes']
        for code in codes:
            if code not in icd10_code_to_group:
                icd10_code_to_group[code] = [group]
            else:
                icd10_code_to_group[code].append(group)

    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

    file_names = []
    loses = []

    ihm_masks = []
    ihm_labels = []
    ihm_positions = []

    los_masks = []
    los_labels = []

    phenotype_labels = []

    decomp_masks = []
    decomp_labels = []

    patients = file_table['patient_id'].unique()
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(root_dir, patient)
        episode_numbers = file_table[file_table['patient_id'] == patient]['episode']
        patient_ts_files = [f"episode{i}_timeseries.csv" for i in episode_numbers]
        stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as ts_file:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file, skip globally
                if label_df.shape[0] == 0:
                    print("\n\t(empty label file)", patient, ts_filename)
                    continue

                # find length of stay, skip globally if it is missing
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                # find all event in ICU, skip globally if there is no event in ICU
                ts_lines = ts_file.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                # In the original code, t was subject to a strict lower bound of -eps to restrict events to
                # the ICU stay. This is not compatible with experiments that require a complete history of
                # events. Therefore, the lower bound has been removed in favor of filtering the events in
                # the upstream extract_episodes_from_subjects.py script as necessary.
                ts_lines = [line for (line, t) in zip(ts_lines, event_times) if t < los + eps]
                event_times = [t for t in event_times if t < los + eps]

                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                # add length of stay
                loses.append(los)

                # find in hospital mortality
                mortality = int(label_df.iloc[0]["Mortality"])

                # write episode data and add file name
                ts_file_path = os.path.abspath(os.path.join(root_dir, patient, ts_filename))
                # if not os.path.exists(os.path.join(output_dir, output_ts_filename)):
                #     with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                #         outfile.write(header)
                #         for line in ts_lines:
                #             outfile.write(line)
                file_names.append(ts_file_path)

                # create in-hospital mortality
                ihm_label = mortality

                ihm_mask = 1
                if los < fixed_hours - eps:
                    ihm_mask = 0
                if event_times[0] > fixed_hours + eps:
                    ihm_mask = 0

                ihm_position = 47
                if ihm_mask == 0:
                    ihm_position = 0

                ihm_masks.append(ihm_mask)
                ihm_labels.append(ihm_label)
                ihm_positions.append(ihm_position)

                # create length of stay
                sample_times = np.arange(0.0, los + eps, sample_rate)
                sample_times = np.array([int(x + eps) for x in sample_times])
                cur_los_masks = map(int, (sample_times > shortest_length) & (sample_times > event_times[0]))
                cur_los_labels = los - sample_times

                los_masks.append(cur_los_masks)
                los_labels.append(cur_los_labels)

                # create phenotyping
                cur_phenotype_labels = [0 for i in range(len(id_to_group))]
                icustay = label_df['Icustay'].iloc[0]
                diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"),
                                           dtype={"ICD_CODE": str, "ICD_VERSION": int})
                diagnoses_df = diagnoses_df[diagnoses_df.ICUSTAY_ID == icustay]

                for index, row in diagnoses_df.iterrows():
                    if row['USE_IN_BENCHMARK']:
                        code = row['ICD_CODE']
                        if row['ICD_VERSION'] == 9:
                            group = icd9_code_to_group[code]
                        elif row['ICD_VERSION'] == 10:
                            group = icd10_code_to_group[code]
                        else:
                            continue
                        for g in group:
                            group_id = group_to_id[g]
                            cur_phenotype_labels[group_id] = 1

                cur_phenotype_labels = [x for (i, x) in enumerate(cur_phenotype_labels)
                                        if definitions[id_to_group[i]]['use_in_benchmark']]
                phenotype_labels.append(cur_phenotype_labels)

                # create decompensation
                stay = stays_df[stays_df.ICUSTAY_ID == icustay]
                deathtime = stay['DEATHTIME'].iloc[0]
                intime = stay['INTIME'].iloc[0]
                if pd.isnull(deathtime):
                    lived_time = 1e18
                else:
                    lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -
                                  datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0

                sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate)
                sample_times = np.array([int(x+eps) for x in sample_times])
                cur_decomp_masks = map(int, (sample_times > shortest_length) & (sample_times > event_times[0]))
                cur_decomp_labels = [(mortality & int(lived_time - t < future_time_interval))
                                     for t in sample_times]
                decomp_masks.append(cur_decomp_masks)
                decomp_labels.append(cur_decomp_labels)

    def permute(arr, p):
        return [arr[index] for index in p]

    if partition == "train":
        perm = list(range(len(file_names)))
        random.shuffle(perm)
    if partition in ("val", "test"):
        perm = list(np.argsort(file_names))

    file_names = permute(file_names, perm)
    loses = permute(loses, perm)

    ihm_masks = permute(ihm_masks, perm)
    ihm_labels = permute(ihm_labels, perm)
    ihm_positions = permute(ihm_positions, perm)

    los_masks = permute(los_masks, perm)
    los_labels = permute(los_labels, perm)

    phenotype_labels = permute(phenotype_labels, perm)

    decomp_masks = permute(decomp_masks, perm)
    decomp_labels = permute(decomp_labels, perm)

    with open(os.path.join(output_dir, f"multitask_{partition}_listfile.csv"), "w") as listfile:
        header = ','.join(['filename', 'length of stay', 'in-hospital mortality task (pos;mask;label)',
                           'length of stay task (masks;labels)', 'phenotyping task (labels)',
                           'decompensation task (masks;labels)'])
        listfile.write(header + "\n")

        for index in range(len(file_names)):
            file_name = file_names[index]
            los = '{:.6f}'.format(loses[index])

            ihm_task = '{:d};{:d};{:d}'.format(ihm_positions[index], ihm_masks[index], ihm_labels[index])

            ls1 = ";".join(map(str, los_masks[index]))
            ls2 = ";".join(map(lambda x: '{:.6f}'.format(x), los_labels[index]))
            los_task = '{};{}'.format(ls1, ls2)

            pheno_task = ';'.join(map(str, phenotype_labels[index]))

            dec1 = ";".join(map(str, decomp_masks[index]))
            dec2 = ";".join(map(str, decomp_labels[index]))
            decomp_task = '{};{}'.format(dec1, dec2)

            listfile.write(','.join([file_name, los, ihm_task, los_task, pheno_task, decomp_task]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create data for multitask prediction.")
    parser.add_argument('root_path', type=str, help="Directory where all patients' timeseries are stored.")
    parser.add_argument('output_path', type=str, help="Directory where the created listfile should be stored. "
                        "Listfiles will be saved in a different subdirectory for each cross validation fold.")
    parser.add_argument('--icd9_phenotype_definitions', '-p9', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                        help='YAML file with phenotype definitions.')
    parser.add_argument('--icd10_phenotype_definitions', '-p10', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccsr_2024_definitions.yaml'),
                        help='YAML file with phenotype definitions.')
    args, _ = parser.parse_known_args()

    with open(args.icd9_phenotype_definitions) as definitions_file:
        icd9_defs = yaml.load(definitions_file, Loader=yaml.Loader)
    with open(args.icd10_phenotype_definitions) as definitions_file:
        icd10_defs = yaml.load(definitions_file, Loader=yaml.Loader)

    definitions = dict()

    for group in icd9_defs:
        if group in definitions.keys():
            definitions[group]['icd9_codes'].extend(icd9_defs[group]['codes'])
            if not definitions[group]['use_in_benchmark']:  # If already True, keep it that way
                definitions[group]['use_in_benchmark'] = icd9_defs[group]['use_in_benchmark']
        else:
            definitions[group] = {
                'icd9_codes': icd9_defs[group]['codes'],
                'icd10_codes': [],
                'use_in_benchmark': icd9_defs[group]['use_in_benchmark']
            }
            
    for group in icd10_defs:
        if group in definitions.keys():
            definitions[group]['icd10_codes'].extend(icd10_defs[group]['codes'])
            if not definitions[group]['use_in_benchmark']:
                definitions[group]['use_in_benchmark'] = icd10_defs[group]['use_in_benchmark']
        else:
            definitions[group] = {
                'icd9_codes': [],
                'icd10_codes': icd10_defs[group]['codes'],
                'use_in_benchmark': icd10_defs[group]['use_in_benchmark']
            }

    resources_dir = get_resources_dir_path()
    partition_table_files = [f for f in os.listdir(resources_dir) if f.startswith('fold')]
    n_folds = max([int(re.match(r'fold(\d+)', f).group(1)) for f in partition_table_files])
    
    for i in range(n_folds + 1):
        print(f"Processing fold {i}...")
        fold_path = os.path.join(args.output_path, f"fold{i}")
        train_file_table = pd.read_csv(
            os.path.join(resources_dir, f'fold{i}_train.csv'), 
            dtype={'patient_id': str, 'episode': int}
        )
        test_file_table = pd.read_csv(
            os.path.join(resources_dir, f'fold{i}_test.csv'),
            dtype={'patient_id': str, 'episode': int}
        )
        if os.path.exists(os.path.join(resources_dir, f'fold{i}_val.csv')):
            val_file_table = pd.read_csv(
                os.path.join(resources_dir, f'fold{i}_val.csv'),
                dtype={'patient_id': str, 'episode': int}
            )
            process_partition(args.root_path, fold_path, "val", val_file_table, definitions)
        process_partition(args.root_path, fold_path, "train", train_file_table, definitions)
        process_partition(args.root_path, fold_path, "test", test_file_table, definitions)
        print('\n')


if __name__ == '__main__':
    main()
