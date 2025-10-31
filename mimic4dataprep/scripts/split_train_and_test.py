"""
This script partitions the data into training and test folds, with the option of setting up folds for K-fold cross 
validation. The file paths to 'episode{i}.csv' files for each fold are written to 'fold{i}_train.csv' and
'fold{i}_test.csv' files in the 'resources' directory after removing any existing tables from previous runs.

Partitioning is done at the patient level to ensure that all episodes from the same patient end up in the same
partition. Command line arguments give the user the option to specify how the episodes are selected from each patient 
for inclusion in a partition. By default, all episodes are included, but the user may specify to include only the 
first, last, or randomly selected episode. The user may also specify whether to stratify the folds by in-hospital 
mortality status. If the all episodes or the last episode is used, in-hospital mortality status is taken from the final
episode. If the first or a random episode is used, the status is taken from the selected episode.
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import numpy as np
import os
import pandas as pd
import re

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from typing import Dict, List

from mimic4dataprep.util import get_resources_dir_path


def remove_existing_partition_tables() -> None:
    """
    Remove existing partition tables from the resources directory.
    """
    resources_dir = get_resources_dir_path()
    to_remove = [f for f in os.listdir(resources_dir) if re.match(r'fold\d+_(train|val|test).csv', f)]
    if len(to_remove) > 0:
        print(f"Removing {len(to_remove)} training/validation/test set partition tables from 'resources' directory.")
        for f in to_remove:
            os.remove(os.path.join(resources_dir, f))


def write_partition_table(
    root_path: str,
    fold: int,
    partition: str,
    patient_ids: List[str],
    patient_episodes: Dict[str, List[int]]
) -> None:
    """
    Write the training, validation, or test partition table for a given fold.

    Args:
        root_path: str
            Path to the root directory containing patient subdirectories.
        fold: int
            Fold number.
        partition: str
            Name of the partition ('train', 'val', or 'test').
        patient_ids: List[str]
            List of IDs of patients in the partition.
        patient_episodes: Dict[str, List[int]]
            Dictionary with patient IDs as keys and lists of episodes to consider during partitioning as values.
    """

    if partition not in ['train', 'val', 'test']:
        raise ValueError(f"'partition' must be one of 'train', 'val', or 'test', got {partition}.")
    
    regex = re.compile(r'/(\d+)/episode(\d+).csv')  # For extracting patient ID and episode number from file path
    
    # Get the file paths for the episodes to include in the partition
    partition_paths = []
    for pt_id in patient_ids:
        patient_path = os.path.abspath(os.path.join(root_path, pt_id))
        episode_file_paths = [os.path.join(patient_path, f'episode{i}.csv') for i in patient_episodes[pt_id]]
        partition_paths.extend(episode_file_paths)
        
    # Extract the the patient IDs and episode numbers from the file path for addition to output table
    partition_table_rows = [(p, *regex.search(p).groups()) for p in partition_paths]
    partition_table = pd.DataFrame(partition_table_rows, columns=['file_path', 'patient_id', 'episode'])
    resources_dir = get_resources_dir_path()
    partition_table.to_csv(os.path.join(resources_dir, f'fold{fold}_{partition}.csv'), index=False)
    

def main():

    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    parser.add_argument('--cv', action='store_true', help='Perform cross-validation.')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument('--make_val_set', action='store_true', help='Create a validation set.')
    parser.add_argument('--stratify_mortality', action='store_true', help='Stratify by in-hospital mortality status.')
    parser.add_argument('--episode_selection', type=str, default='all', 
                        help='How episodes are selected from each patient for inclusion in the fold. Options: last '
                        '(default), first, random, all.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args, _ = parser.parse_known_args()

    remove_existing_partition_tables()

    rng = np.random.default_rng(args.seed)

    all_patient_ids = []
    mortality = []
    patient_episodes = dict()  # Key: patient ID, Value: list of episodes to consider during partitioning

    # Get list of all patient directories in subjects_root_path
    all_patient_ids = [d for d in os.listdir(args.subjects_root_path) 
                       if os.path.isdir(os.path.join(args.subjects_root_path, d))]
    
    regex = re.compile(r'episode(\d+).csv')

    for pt_id in all_patient_ids.copy():
        # print(pt_id)
        patient_path = os.path.join(args.subjects_root_path, pt_id)
        files = os.listdir(patient_path)
        all_episodes = [int(regex.search(x).group(1)) for x in files if regex.match(x)]  # Extract episode numbers
        if len(all_episodes) == 0:
            print(f"No episodes found for patient {pt_id}. Skipping.")
            all_patient_ids.remove(pt_id)  # Remove patient ID from list if no episodes found
            continue
        # Select episodes to consider during data partitioning and get the correct episode for mortality status
        if args.episode_selection == 'all':
            mortality_episode = max(all_episodes)
            episodes_used = all_episodes
        elif args.episode_selection == 'last':
            mortality_episode = max(all_episodes)
            episodes_used = [mortality_episode]
        elif args.episode_selection == 'first':
            mortality_episode = min(all_episodes)
            episodes_used = [mortality_episode]
        else:  # Randomly select one episode per patient
            mortality_episode = rng.integers(1, len(all_episodes) + 1, size=1)[0]
            episodes_used = [mortality_episode]
        patient_episodes[pt_id] = episodes_used  # Update dict with a list of episodes to consider for this patient

        if args.stratify_mortality:
            # Append to the list of labels (in-hospital mortality status) used for stratified k-fold CV
            fp = os.path.join(args.subjects_root_path, pt_id, f'episode{mortality_episode}.csv')
            mortality.append(int(pd.read_csv(fp, usecols=['Mortality']).iloc[0, 0]))
    
    if args.cv:  # Partition data for cross-validation

        if args.stratify_mortality:
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            for fold, (train_idx, test_idx) in enumerate(skf.split(all_patient_ids, mortality)):
                train_pts = [all_patient_ids[i] for i in train_idx]
                test_pts = [all_patient_ids[i] for i in test_idx]
                if args.make_val_set:
                    train_pts, val_pts = train_test_split(
                        train_pts, test_size=0.2, random_state=args.seed, stratify=[mortality[i] for i in train_idx]
                    )
                    write_partition_table(args.subjects_root_path, fold, 'val', val_pts, patient_episodes)
                write_partition_table(args.subjects_root_path, fold, 'train', train_pts, patient_episodes)
                write_partition_table(args.subjects_root_path, fold, 'test', test_pts, patient_episodes)
        else:
            kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            for fold, (train_idx, test_idx) in enumerate(kf.split(all_patient_ids)):
                train_pts = [all_patient_ids[i] for i in train_idx]
                test_pts = [all_patient_ids[i] for i in test_idx]
                if args.make_val_set:
                    train_pts, val_pts = train_test_split(train_pts, test_size=0.2, random_state=args.seed)
                    write_partition_table(args.subjects_root_path, fold, 'val', val_pts, patient_episodes)
                write_partition_table(args.subjects_root_path, fold, 'train', train_pts, patient_episodes)
                write_partition_table(args.subjects_root_path, fold, 'test', test_pts, patient_episodes)
                                       
    else:  # Split data into singular train and test sets
 
        if args.stratify_mortality:
            train_pts, test_pts = train_test_split(
                all_patient_ids, test_size=0.2, random_state=args.seed, stratify=mortality
            )
            if args.make_val_set:
                train_pts, val_pts = train_test_split(
                    train_pts, test_size=0.2, random_state=args.seed, 
                    stratify=[mortality[all_patient_ids.index(pt)] for pt in train_pts]
                )
                write_partition_table(args.subjects_root_path, 0, 'val', val_pts, patient_episodes)
        else:
            train_pts, test_pts = train_test_split(
                all_patient_ids, test_size=0.2, random_state=args.seed
            )
            if args.make_val_set:
                train_pts, val_pts = train_test_split(
                    train_pts, test_size=0.2, random_state=args.seed
                )
                write_partition_table(args.subjects_root_path, 0, 'val', val_pts, patient_episodes)

        # Create partitions under a fold1 directory to simplify the directory structure for downstream scripts
        write_partition_table(args.subjects_root_path, 0, 'train', train_pts, patient_episodes)
        write_partition_table(args.subjects_root_path, 0, 'test', test_pts, patient_episodes)


if __name__ == '__main__':
    main()
