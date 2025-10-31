# MIMIC-IV Benchmarks

## About

This repository is a modification of the work published by [Harutyunyan *et al*.](https://github.com/YerevaNN/mimic3-benchmarks/tree/v1.0.0-alpha) (see the associated [publication](https://www.nature.com/articles/s41597-019-0103-9)). The original repository offers a collection of Python tools for benchmarking medical prediction tasks using ICU stay data from the [Medical Information Mart for Intensive Care (MIMIC) III](https://www.nature.com/articles/sdata201635) database.

This repository adapts the original to work with a newer version of the MIMIC dataset, [MIMIC-IV](https://www.nature.com/articles/s41597-022-01899-x). Several other modifications have also been made; see the "Major Changes" section further below. But first...

## A Very Important Note

The changes made in this repository only extend to dataset creation. There have been no modifications to the original modeling routines, making them incompatible with the new dataset creation routines. I *do not* plan to adapt the modeling routines in the future.

Modifications apply to the following scripts and their local import chains:

- `mimic4dataprep/scripts/extract_subjects.py`
- `mimic4dataprep/scripts/validate_events.py`
- `mimic4dataprep/scripts/extract_episodes_from_subjects.py`
- `mimic4dataprep/scripts/split_train_and_test.py`
- `mimic4dataprep/scripts/create_decompensation.py`
- `mimic4dataprep/scripts/create_in_hospital_mortality.py`
- `mimic4dataprep/scripts/create_length_of_stay.py`
- `mimic4dataprep/scripts/create_phenotyping.py`
- `mimic4dataprep/scripts/create_multitask.py`
- `mimic4dataprep/readers.py`

## Major Changes

- **Now supports MIMIC-IV**. MIMIC-III is not supported.

- **Labels for the phenotype prediction task are now derived from both ICD-9 and ICD-10 diagnosis codes.** Formerly, only ICD-9 codes were supported. Categories for phenotype prediction were originally derived from the 2015 Healthcare Cost & Utilization Project Clinical Classifications Software ([HCUP CCS](https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)), which maps ICD-9 codes to the categories. I've added support for ICD-10 codes, which map to categories according to the 2024 revision of the classifications software ([HCUP CCSR 2024.1](https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccsr_archive.jsp#ccsr)). Note that ICD-10 codes may map to more than one category whereas the ICD-9 codes only mapped to a single category. Also note that the category description differ slightly between HCUP CCS and HCUP CCSR, and I have matched/renamed the categories in HCUP CCSR to match the 25 categories of interest in the original multilabel phenotype prediction task. If you're interested in adding more categories to the phenotype prediction task, you'll have to make sure the descriptions align for whichever new categories you add.

- **Added the option to create timeseries of patient records that span the patient's entire history of medical records through the end of the ICU stay in question**. Formerly, the scripts only created timeseries data from records generated during the ICU stay. This is still possible in the revised code, but now you have the option to consider the complete history by setting `--use_full_record_history` flags when running applicable scripts.

- **Added the option to include discharge summary text as a feature**. The discharge summaries are obtained from the [MIMIC-IV-Note](https://www.physionet.org/content/mimic-iv-note/2.2/) database, which is separate from the main MIMIC-IV database. This option can be exercised by providing a path to the notes `discharge.csv` with `--notes_path` when running the `extract_subjects` script. Note that you *must* set the `--use_full_record_history` flags to use discharge summaries (discharge summaries from the hospitalization associated with the current ICU stay are future events and won't be part of the timeseries of records from the ICU stay).

- **Added support for K-fold cross validation, with the option to stratify by in-hospital mortality**. The original code did not support cross-validation, nor did it support stratification. Stratification ensures that the training, validation, and test sets have roughly the same proportion of in-hospital mortalities across all folds.

- **Reduced the on-disk memory footprint of training, validation, and test set data creation**. Previously, the `split_train_and_test` script moved files from the root data directory to training and test set directories. The dataset creation scripts for each benchmark prediciton task would then create new dataset files that replicated existing on-disk data several times over. If this strategy is used with cross-validation, the memory footprint is huge. In the revised version of `split_train_and_test`, lists of paths to patient-episode timeseries in the training, test, and (optionally) validation sets are written to the `resources` directory. The `Reader` classes have been adapted to use these lists to locate the original timeseries csv and read data from it directly, eliminating the need to create new csv files on disk. This significantly limits the on-disk memory footprint.

## Structure

There are four parts in this repository. 

* Tools for creating the benchmark datasets.  
* Tools for reading the benchmark datasets.
* Evaluation scripts.
* Baseline models and helper tools.

Only the first dataset creation and reading parts have been modified from the originals. The modeling and evaluation scripts have *not* been adapted for use with the new dataset creation framework.

The `mimic4dataprep/scripts` directory contains scripts for creating the benchmark datasets.
The reading tools are in `mimic4dataprep/readers.py`.

## Requirements

See `requirements.txt`. For dataset creation and reading scripts only `numpy`, `pandas`, and `scikit-learn` are necessary. `Keras` is only used by modeling scripts.

The MIMIC data are sensitive and require special authorization to access. If you already have access privileges, you can find the MIMIC-IV dataset [here](https://physionet.org/content/mimiciv/3.1/). The MIMIC-IV-Note dataset can be found [here](https://www.physionet.org/content/mimic-iv-note/2.2/).

## Creating a Benchmark Dataset

1. Download the MIMIC-IV data and, optionally, MIMIC-IV-Note data.

2. Clone the required repos:

        ```shell
        git clone https://github.com/mdparkes/mimic4dataprep/ && cd mimic4dataprep/
        git clone https://github.com/mdparkes/datacleaner/ && pip install ./datacleaner
        ```

3. Extract patients' medical record data from the downloaded MIMIC-IV csv files, e.g.:

        ```shell
        python -m mimic4dataprep.scripts.extract_subjects ./mimic-iv data/root/ --event_tables chartevents labevents outputevents discharge --notes_path ./mimic-iv-note/2.2/note
        ```
    
    In the example above, the `extract_subjects` creates patient subdirectories in `data/root` and populates them with information about ICU stays (`stays.csv`), diagnoses (`diagnoses.csv`), and timestamped events in the patient's complete history of medical records (`events.csv`). By default, `stays.csv` only includes ICU stays that occurred during hospitalizations with a single ICU stay with no transfers between wards.

    Setting `--notes_path` is optional and only needed if you wish to use discharge summary text as a feature in the datasets. If so, you must use the `--use_full_record_history` flag in the `validate_events` and `extract_episodes_from_subjects` scripts.

4. Validate the extracted data, e.g.:

        ```shell
        python -m mimic4dataprep.scripts.validate_events data/root/ --use_full_record_history
        ```

    The --use_full_record_history flag is optional and will perform different data validation steps if used. See the docstring in the script for more details. If not used, the script will perform the steps originally used by Harutyunyan *et al*., which includes removing events with missing crucial information and attempting to infer missing IDs (see also `more_on_validating_events.md`). The script is designed for use only with `stays.csv` files that were generated to include hospital admissions with only a single ICU stay without transfers between wards.

5. Create data files that break up `events.csv` by ICU stay episodes:

        ```shell
        python -m mimic4dataprep.scripts.extract_episodes_from_subjects data/root/ --use_full_record_history
        ```

    In this example, the extracted timeseries are written to disk as `/data/root/{SUBJECT_ID}/episode{i}_timeseries.csv`, where *i* denotes a specific ICU stay extracted from the `events.csv` file. Again, `--use_full_record_history` is optional, and necessary if you want to include discharge summaries from MIMIC-IV-Note as features. If set, the *i*th episode's timeseries will include events from all previous hospitalizations and ICU stays through the end of the *i*th ICU stay.


6. Split the dataset into training, test, and (optionally) validation sets, with the option to create stratified or unstratified folds for cross validation, e.g.:

        ```shell
        python -m mimic4dataprep.scripts.split_train_and_test data/root/ --cv --n_folds 5 --stratify_mortality --make_val_set --episode_selection last
        ```

    Training/test set splits are 80%/20%. If the `--cv` flag is not set, a simple split is performed. If `--make_val_set` is used, validation sets are created from 20% of the data in the training set resulting in 64% of the data being set aside fo training, 16% for validation, and 20% for testing. Data are partitioned at the patient level to ensure that ICU stay episodes from the same patient never occupy both the training and validation or test sets.  The `--episode_selection` parameter controls which of each patient's ICU stay episodes are included in the dataset that will be split. It will accept `first`, `last`, `random`, or `all` as an argument. To stratify the partitions by in-hospital mortality status, use the `--stratify_mortality` flag. If `--episode_selection` is `last` or `all`, the mortality status from the patient's final documented ICU stay episode will be used for stratification. Otherwise, the patient's mortality status will be taken from the selected episode. This script will write csv files to the `resources` folder specifying the file paths to timeseries data for all the episodes in each data partition. 

7. Create task-specific datasets for benchmark prediction tasks:

        ```shell
        python -m mimic4dataprep.scripts.create_in_hospital_mortality data/root/ data
        python -m mimic4dataprep.scripts.create_decompensation data/root/ data
        python -m mimic4dataprep.scripts.create_length_of_stay data/root/ data
        python -m mimic4dataprep.scripts.create_phenotyping data/root/ data
        python -m mimic4dataprep.scripts.create_multitask data/root/ data
        ```
    In this example the scripts will create csv listfiles for the training/validation/test partitions in each fold under `/data/fold{i}`. The listfiles specify the file paths to timeseries data for episodes in the partition, labels for the benchmark prediction task, and their timestamps. The assumption is that all benchmark prediction tasks will use the same training/validation/test partitions.

## Readers

Instances of the `Reader` class are meant to help you read timeseries data. Each benchmark prediction task is associated with its own subclass of `Reader`. Instances of `Reader` use the listfiles created in step 6 above to find the on-disk timeseries data for each observation and return it together with the label and its timestamp. For more information on readers see [`mimic4dataprep/more_on_readers.md`](mimic4dataprep/more_on_readers.md).
