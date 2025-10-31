from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
import re

from tqdm import tqdm

from mimic4dataprep.util import get_resources_dir_path

random.seed(49297)


def process_partition(root_dir, output_dir, partition, file_table, eps=1e-6, n_hours=48):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xy_pairs = []
    patients = file_table['patient_id'].unique()
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(root_dir, patient)
        episode_numbers = file_table[file_table['patient_id'] == patient]['episode']
        patient_ts_files = [f"episode{i}_timeseries.csv" for i in episode_numbers]

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                # In the original code, t was subject to a strict lower bound of -eps to restrict events to
                # the ICU stay. This is not compatible with experiments that require a complete history of
                # events. Therefore, the lower bound has been removed in favor of filtering the events in
                # the upstream extract_episodes_from_subjects.py script as necessary.
                ts_lines = [line for (line, t) in zip(ts_lines, event_times) if t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                ts_file_path = os.path.abspath(os.path.join(root_dir, patient, ts_filename))
                # if not os.path.exists(os.path.join(output_dir, output_ts_filename)):
                #     with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                #         outfile.write(header)
                #         for line in ts_lines:
                #             outfile.write(line)

                xy_pairs.append((ts_file_path, mortality))

    if partition == "train":
        random.shuffle(xy_pairs)
    if partition in ("val", "test"):
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, f"in_hospital_mortality_{partition}_listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in xy_pairs:
            listfile.write('{},{:d}\n'.format(x, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Directory where all patients' timeseries are stored.")
    parser.add_argument('output_path', type=str, help="Directory where the created listfile should be stored. "
                        "Listfiles will be saved in a different subdirectory for each cross validation fold.")
    args, _ = parser.parse_known_args()

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
            process_partition(args.root_path, fold_path, "val", val_file_table)
        process_partition(args.root_path, fold_path, "train", train_file_table)
        process_partition(args.root_path, fold_path, "test", test_file_table)
        print('\n')


if __name__ == '__main__':
    main()
