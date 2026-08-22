"""
Audit the alignment between the HCUP CCS 2015 (ICD-9-CM) and HCUP CCSR 2024.1 (ICD-10-CM) category definitions
that back the 25 benchmark phenotypes.

The two taxonomies are reconciled by renaming CCSR categories to their CCS counterparts in
resources/dump_hcup_ccsr_yaml.py, so that one phenotype occupies one column of the label matrix regardless of the
coding era of the stay. When a rename is missing or a CCSR subdivision is left out, the resulting label is
confounded with the admission date: it can only ever be positive for stays coded in one of the two eras. This
script detects that failure mode.

The code-coverage audit needs no patient data. Pass --diagnoses to additionally compare observed per-phenotype
prevalence between eras.
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys

import pandas as pd
import yaml

import mimic4dataprep
from mimic4dataprep.preprocessing import make_phenotype_label_matrix


RESOURCES = os.path.join(os.path.dirname(os.path.abspath(mimic4dataprep.__file__)), 'resources')


def read_definitions(file_path):
    with open(file_path, 'r') as f_in:
        return yaml.safe_load(f_in)


def benchmark_code_sets(definitions):
    return {
        category: set(defn['codes']) for category, defn in definitions.items() if defn['use_in_benchmark']
    }


def audit_code_coverage(icd9_definitions, icd10_definitions):
    """
    Count the ICD-9-CM and ICD-10-CM codes backing each benchmark phenotype.

    A benchmark category with no codes in one era cannot be positive for any stay coded in that era, which makes
    the label a proxy for the admission date rather than for the patient's condition.
    """
    icd9 = benchmark_code_sets(icd9_definitions)
    icd10 = benchmark_code_sets(icd10_definitions)
    audit = pd.DataFrame([
        {
            'CATEGORY': category,
            'N_ICD9_CODES': len(icd9.get(category, ())),
            'N_ICD10_CODES': len(icd10.get(category, ()))
        }
        for category in sorted(set(icd9) | set(icd10))
    ])
    audit['ERA_SPECIFIC'] = (audit['N_ICD9_CODES'] == 0) | (audit['N_ICD10_CODES'] == 0)
    return audit


def assign_coding_era(diagnoses):
    """
    Label each ICU stay with the ICD version used for the majority of its diagnosis codes. Stays carrying codes
    from both versions are reported separately, since their labels draw on both definition files.
    """
    counts = diagnoses.groupby(['ICUSTAY_ID', 'ICD_VERSION']).size().unstack(fill_value=0)
    counts.columns = counts.columns.astype(int)
    era = counts.idxmax(axis=1).rename('ICD_VERSION')
    mixed = (counts > 0).sum(axis=1) > 1
    return era, mixed


def audit_era_prevalence(diagnoses, icd9_definitions, icd10_definitions):
    """
    Compare per-phenotype prevalence between ICU stays coded in ICD-9-CM and those coded in ICD-10-CM.

    Prevalence legitimately drifts between eras because of changes in case mix and coding practice, so a
    difference here is not on its own an error. A phenotype that is common in one era and near-absent in the
    other is the signature of a category alignment problem.
    """
    labels = make_phenotype_label_matrix(diagnoses, icd9_definitions, icd10_definitions)
    era, mixed = assign_coding_era(diagnoses)
    era = era.reindex(labels.index)
    prevalence = labels.groupby(era).mean().T
    prevalence.columns = ['PREV_ICD%d' % v for v in prevalence.columns]
    prevalence.index.name = 'CATEGORY'
    stay_counts = era.value_counts().sort_index()
    return prevalence.reset_index(), stay_counts, int(mixed.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--icd9-definitions', default=os.path.join(RESOURCES, 'hcup_ccs_2015_definitions.yaml'),
                        help='Path to the HCUP CCS 2015 definitions YAML file.')
    parser.add_argument('--icd10-definitions', default=os.path.join(RESOURCES, 'hcup_ccsr_2024_definitions.yaml'),
                        help='Path to the HCUP CCSR 2024.1 definitions YAML file.')
    parser.add_argument('--diagnoses', default=None,
                        help='Path to all_diagnoses.csv. When given, per-era phenotype prevalence is also '
                             'reported.')
    parser.add_argument('--output', default=None, help='Write the audit table to this CSV file.')
    args = parser.parse_args()

    icd9_definitions = read_definitions(args.icd9_definitions)
    icd10_definitions = read_definitions(args.icd10_definitions)
    audit = audit_code_coverage(icd9_definitions, icd10_definitions)

    if args.diagnoses is not None:
        diagnoses = pd.read_csv(args.diagnoses, dtype={'ICD_CODE': str})
        prevalence, stay_counts, n_mixed = audit_era_prevalence(diagnoses, icd9_definitions, icd10_definitions)
        audit = audit.merge(prevalence, on='CATEGORY', how='left')
        print('ICU stays by coding era:')
        print(stay_counts.to_string(), '\n')
        if n_mixed:
            print('%d stays carry both ICD-9-CM and ICD-10-CM codes.\n' % n_mixed)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 64)
    print(audit.to_string(index=False), '\n')
    print('%d benchmark phenotypes' % len(audit))

    if args.output is not None:
        audit.to_csv(args.output, index=False)

    era_specific = audit.loc[audit['ERA_SPECIFIC'], 'CATEGORY'].tolist()
    if era_specific:
        print('\nFAIL: %d phenotype(s) are defined for only one coding era:' % len(era_specific))
        for category in era_specific:
            print('  %s' % category)
        print('Add the missing CCSR/CCS rename in resources/dump_hcup_ccsr_yaml.py and regenerate the YAML file.')
        return 1
    print('OK: every benchmark phenotype is defined for both coding eras.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
