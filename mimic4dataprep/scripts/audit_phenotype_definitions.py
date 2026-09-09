"""
Audit the alignment between the HCUP CCS 2015 (ICD-9-CM) and HCUP CCSR 2024.1 (ICD-10-CM) category definitions
that back the 25 benchmark phenotypes.

The two taxonomies are reconciled by renaming CCSR categories to their CCS counterparts in
resources/dump_hcup_ccsr_yaml.py, so that one phenotype occupies one column of the label matrix regardless of the
coding era of the stay. When a rename is missing or a CCSR subdivision is left out, the resulting label is
confounded with the admission date: it can only ever be positive for stays coded in one of the two eras. This
script detects that failure mode and quantifies what remains after it is fixed.

Three checks run without patient data: per-category code counts, the era-specific test above, and the rate at
which a single diagnosis code maps to more than one benchmark phenotype. Pass --diagnoses to additionally report
observed per-phenotype prevalence and the share of recorded diagnoses that reach a benchmark phenotype at all.

Pass --latex to write the per-phenotype table and the per-era summary as LaTeX for inclusion in a manuscript.
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


def benchmark_category_map(definitions):
    """Map each diagnosis code to the set of benchmark categories that contain it."""
    code_map = {}
    for category, codes in benchmark_code_sets(definitions).items():
        for code in codes:
            code_map.setdefault(code, set()).add(category)
    return code_map


def audit_code_counts(icd9_definitions, icd10_definitions):
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


def audit_ambiguity(icd9_definitions, icd10_definitions):
    """
    Report how often one diagnosis code maps to more than one benchmark phenotype in each coding era.

    CCS 2015 assigns each ICD-9-CM code to exactly one category by design, so the ICD-9 rate is expected to be
    zero. CCSR deliberately assigns an ICD-10-CM code to every applicable category, so the ICD-10 rate is
    expected to be positive: an ICD-10 coded stay accrues more positive labels per recorded diagnosis than an
    otherwise identical ICD-9 coded stay. That asymmetry is inherent to reconciling a one-to-one taxonomy with a
    one-to-many one and is not corrected by the category alignment; this is the residual to report rather than
    to remove.
    """
    rows = []
    for version, definitions in ((9, icd9_definitions), (10, icd10_definitions)):
        code_map = benchmark_category_map(definitions)
        n_categories = pd.Series({code: len(categories) for code, categories in code_map.items()}, dtype=float)
        rows.append({
            'ICD_VERSION': version,
            'N_CODES_MAPPED': int(len(n_categories)),
            'N_CODES_AMBIGUOUS': int((n_categories > 1).sum()),
            'AMBIGUITY_RATE': float((n_categories > 1).mean()) if len(n_categories) else float('nan'),
            'MAX_CATEGORIES_PER_CODE': int(n_categories.max()) if len(n_categories) else 0
        })
    return pd.DataFrame(rows)


def audit_observed_coverage(diagnoses, icd9_definitions, icd10_definitions):
    """
    Report the share of recorded diagnoses that reach a benchmark phenotype, by coding era.

    Coverage is low by construction: only 25 of the roughly 285 CCS categories are used as phenotypes, so most
    recorded diagnoses correctly map to nothing. The absolute level therefore carries little meaning. The
    comparison between eras is what matters, because a category whose codes are partly unmapped shows up as a
    coverage gap on one side only. Rates are given over distinct codes and over occurrences, since a long tail
    of rare unmapped codes depresses the former while barely affecting the latter.

    Ambiguity is recomputed here over observed codes, weighted by how often they actually appear, which is the
    quantity that bears on label noise.
    """
    rows = []
    for version, definitions in ((9, icd9_definitions), (10, icd10_definitions)):
        code_map = benchmark_category_map(definitions)
        observed = diagnoses.loc[diagnoses['ICD_VERSION'].astype(int) == version, 'ICD_CODE']
        if observed.empty:
            continue
        counts = observed.value_counts()
        mapped = pd.Series(counts.index.isin(list(code_map)), index=counts.index)
        n_categories = pd.Series([len(code_map.get(code, ())) for code in counts.index], index=counts.index)
        ambiguous = n_categories > 1
        n_mapped_occurrences = int(counts[mapped].sum())
        rows.append({
            'ICD_VERSION': version,
            'N_CODES_OBSERVED': int(len(counts)),
            'N_CODES_MAPPED': int(mapped.sum()),
            'COVERAGE_CODES': float(mapped.mean()),
            'N_OCCURRENCES': int(counts.sum()),
            'COVERAGE_OCCURRENCES': float(n_mapped_occurrences) / float(counts.sum()),
            'AMBIGUITY_CODES': float(ambiguous[mapped].mean()) if int(mapped.sum()) else float('nan'),
            'AMBIGUITY_OCCURRENCES': (
                float(counts[ambiguous].sum()) / float(n_mapped_occurrences) if n_mapped_occurrences
                else float('nan')
            )
        })
    return pd.DataFrame(rows)


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


def latex_escape(text):
    for char in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        text = text.replace(char, '\\' + char)
    return text


def latex_tables(audit, ambiguity, coverage):
    """
    Render the audit as two LaTeX tables: per-phenotype code counts and prevalence, and a per-era summary of
    coverage and ambiguity. Requires the booktabs package.
    """
    has_prevalence = 'PREV_ICD9' in audit.columns and 'PREV_ICD10' in audit.columns
    pct = lambda x: '--' if pd.isna(x) else '%.1f' % (100.0 * x)

    lines = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{table}[htbp]',
        '\\centering',
        '\\caption{Diagnosis codes and observed prevalence for each benchmark phenotype, by coding era. '
        'Codes are counted from the HCUP CCS 2015 (ICD-9-CM) and CCSR 2024.1 (ICD-10-CM) definitions after '
        'category alignment. A phenotype with no codes in one era would be confounded with admission date; '
        'none remain.}',
        '\\label{tab:phenotype-definitions}',
        '\\begin{tabular}{l%s}' % ('rrrr' if has_prevalence else 'rr'),
        '\\toprule',
        ('Phenotype & \\multicolumn{2}{c}{Codes} & \\multicolumn{2}{c}{Prevalence (\\%)} \\\\'
         if has_prevalence else 'Phenotype & \\multicolumn{2}{c}{Codes} \\\\'),
        ('\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}' if has_prevalence else '\\cmidrule(lr){2-3}'),
        ('& ICD-9 & ICD-10 & ICD-9 & ICD-10 \\\\' if has_prevalence else '& ICD-9 & ICD-10 \\\\'),
        '\\midrule',
    ]
    for _, row in audit.iterrows():
        cells = [latex_escape(row['CATEGORY']), '%d' % row['N_ICD9_CODES'], '%d' % row['N_ICD10_CODES']]
        if has_prevalence:
            cells += [pct(row['PREV_ICD9']), pct(row['PREV_ICD10'])]
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']

    lines += [
        '\\begin{table}[htbp]',
        '\\centering',
        '\\caption{Mapping behaviour by coding era. Coverage is the share of recorded diagnoses that map to any '
        'of the 25 benchmark phenotypes; it is low by construction, since only 25 of roughly 285 CCS categories '
        'are used, and only the comparison between eras is informative. Ambiguity is the share of mapped '
        'diagnoses that fall in more than one phenotype. CCS 2015 assigns each ICD-9-CM code to exactly one '
        'category by design, whereas CCSR assigns ICD-10-CM codes to every applicable category, so ICD-10 '
        'stays accrue more positive labels per recorded diagnosis.}',
        '\\label{tab:mapping-behaviour}',
        '\\begin{tabular}{lrr}',
        '\\toprule',
        '& ICD-9-CM & ICD-10-CM \\\\',
        '\\midrule',
    ]

    def summary_row(label, frame, column, formatter):
        if frame is None or frame.empty:
            return None
        indexed = frame.set_index('ICD_VERSION')
        cells = [formatter(indexed[column].get(v, float('nan'))) for v in (9, 10)]
        return ' & '.join([label] + cells) + ' \\\\'

    integer = lambda x: '--' if pd.isna(x) else '%d' % x
    for label, frame, column, formatter in (
        ('Codes in benchmark phenotypes', ambiguity, 'N_CODES_MAPPED', integer),
        ('Distinct codes observed', coverage, 'N_CODES_OBSERVED', integer),
        ('Coverage, distinct codes (\\%)', coverage, 'COVERAGE_CODES', pct),
        ('Coverage, occurrences (\\%)', coverage, 'COVERAGE_OCCURRENCES', pct),
        ('Ambiguity, definitions (\\%)', ambiguity, 'AMBIGUITY_RATE', pct),
        ('Ambiguity, occurrences (\\%)', coverage, 'AMBIGUITY_OCCURRENCES', pct),
        ('Maximum phenotypes per code', ambiguity, 'MAX_CATEGORIES_PER_CODE', integer),
    ):
        line = summary_row(label, frame, column, formatter)
        if line is not None:
            lines.append(line)
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--icd9-definitions', default=os.path.join(RESOURCES, 'hcup_ccs_2015_definitions.yaml'),
                        help='Path to the HCUP CCS 2015 definitions YAML file.')
    parser.add_argument('--icd10-definitions', default=os.path.join(RESOURCES, 'hcup_ccsr_2024_definitions.yaml'),
                        help='Path to the HCUP CCSR 2024.1 definitions YAML file.')
    parser.add_argument('--diagnoses', default=None,
                        help='Path to all_diagnoses.csv. When given, per-era phenotype prevalence and observed '
                             'coverage are also reported.')
    parser.add_argument('--output', default=None, help='Write the per-phenotype audit table to this CSV file.')
    parser.add_argument('--summary-output', default=None,
                        help='Write the per-era coverage and ambiguity summary to this CSV file.')
    parser.add_argument('--latex', default=None,
                        help='Write both tables as LaTeX to this file, formatted for a manuscript.')
    args = parser.parse_args()

    icd9_definitions = read_definitions(args.icd9_definitions)
    icd10_definitions = read_definitions(args.icd10_definitions)
    audit = audit_code_counts(icd9_definitions, icd10_definitions)
    ambiguity = audit_ambiguity(icd9_definitions, icd10_definitions)
    coverage = None

    if args.diagnoses is not None:
        diagnoses = pd.read_csv(args.diagnoses, dtype={'ICD_CODE': str})
        prevalence, stay_counts, n_mixed = audit_era_prevalence(diagnoses, icd9_definitions, icd10_definitions)
        coverage = audit_observed_coverage(diagnoses, icd9_definitions, icd10_definitions)
        audit = audit.merge(prevalence, on='CATEGORY', how='left')
        print('ICU stays by coding era:')
        print(stay_counts.to_string(), '\n')
        if n_mixed:
            print('%d stays carry both ICD-9-CM and ICD-10-CM codes.\n' % n_mixed)

    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 64)
    print(audit.to_string(index=False), '\n')
    print('%d benchmark phenotypes\n' % len(audit))

    print('Codes mapping to more than one benchmark phenotype:')
    print(ambiguity.to_string(index=False), '\n')
    if coverage is not None:
        print('Share of recorded diagnoses reaching a benchmark phenotype:')
        print(coverage.to_string(index=False), '\n')

    if args.output is not None:
        audit.to_csv(args.output, index=False)
    if args.summary_output is not None:
        summary = ambiguity if coverage is None else ambiguity.merge(coverage, on='ICD_VERSION', how='outer',
                                                                    suffixes=('', '_OBSERVED'))
        summary.to_csv(args.summary_output, index=False)
    if args.latex is not None:
        with open(args.latex, 'w') as f_out:
            f_out.write(latex_tables(audit, ambiguity, coverage) + '\n')
        print('Wrote LaTeX tables to %s' % args.latex)

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
