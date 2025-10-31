from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import pandas as pd
import yaml

from pandas import DataFrame
from typing import Optional

from mimic4dataprep.util import dataframe_from_csv

###############################
# Non-time series preprocessing
###############################

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}


# NOTE The ethnicity map has been redone because the original mapping didn't make much sense. For instance, white, 
# middle eastern, and portuguese (for some reason) were all lumped together, as were american indian, native hawaiian, 
# and other/unknown. Racial differences in health outcomes relates to more than just how a person looks, it has to do
# with genetic differences between populations as well.
e_map = {
    'ASIAN': 1,
    'BLACK': 2,
    'CARIBBEAN ISLAND': 3,
    'HISPANIC': 4,
    'SOUTH AMERICAN': 5,
    'WHITE': 6,
    'MIDDLE EASTERN': 7,
    'AMERICAN INDIAN': 8,
    'NATIVE HAWAIIAN': 9,  # i.e. Polynesian/pacific islander
    'PORTUGUESE': 0,  # This doesn't reveal ethnicity, so treat it as unknown
    'UNABLE TO OBTAIN': 0,
    'PATIENT DECLINED TO ANSWER': 0,
    'UNKNOWN': 0,
    'OTHER': 0,
    '': 0
}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}


def assemble_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.ICUSTAY_ID, 'Age': stays.AGE, 'Length of Stay': stays.LOS,
            'Mortality': stays.MORTALITY}
    data.update(transform_gender(stays.GENDER))
    data.update(transform_ethnicity(stays.ETHNICITY))
    data['Height'] = np.nan
    data['Weight'] = np.nan
    data = DataFrame(data).set_index('Icustay')
    data = data[['Ethnicity', 'Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']]
    return data.merge(extract_diagnosis_labels(diagnoses, diagnosis_labels), left_index=True, right_index=True)

# diagnosis_labels = ['4019', '4280', '41401', '42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720',
#                     '2859', '2449', '486', '2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070',
#                     '40390', '3051', '412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867',
#                     '4241', '40391', '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749',
#                     '4168', '5180', '45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111',
#                     'V1251', '30000', '3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652',
#                     '99811', '431', '28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832',
#                     'E8788', '00845', '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703',
#                     '2809', '5712', '27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254',
#                     '42823', '28529', 'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859',
#                     'V667', 'E8497', '79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770',
#                     'V5865', '99662', '28860', '36201', '56210']


def get_diagnosis_labels() -> list[str]:
    """
    Extract ICD-9-CM and ICD-10-CM codes that fall under benchmark HCUP CCS/CCSR categories for phenotype 
    classification.
    
    Note that Harutyunyan et al. originally hard-coded the ICD-9 codes to use as diagnosis labels (see commented out
    code above), but their hard-coded selection of codes wasn't explained in the paper. This function is consistent
    with the methods described in the paper, but results in a larger collection of ICD codes than what was hard-coded
    above by Harutyunyan et al. This is not just because it includes both ICD9 and ICD10 codes, but also because the
    hard-coded list of ICD9 codes did not fully cover all the HCUP CCS ccategories that are used in the benchmark.
    """
    
    benchmarks = set()
    root_path = os.path.dirname(__file__)
    
    # ICD-9-CM codes are mapped to categories in the 2015 HCUP CCS
    fp = os.path.join(root_path, f'resources/hcup_ccs_2015_definitions.yaml')
    with open(fp, 'r') as f_in:
        definitions = yaml.safe_load(f_in)
    for defn in definitions.values():
        if defn['use_in_benchmark']:
            benchmarks.update(set(defn['codes']))
    
    # ICD-10-CM codes are mapped to categories in the 2024 HCUP CCSR
    fp = os.path.join(root_path, f'resources/hcup_ccsr_2024_definitions.yaml')
    with open(fp, 'r') as f_in:
        definitions = yaml.safe_load(f_in)
    for defn in definitions.values():
        if defn['use_in_benchmark']:
            benchmarks.update(set(defn['codes']))
    return sorted(list(benchmarks))


diagnosis_labels = get_diagnosis_labels()


def extract_diagnosis_labels(diagnoses, diagnosis_labels):
    """
    This function returns a DataFrame indicating which diagnoses in diagnosis_labels were recorded in the hospital
    stay of each ICU stay. The rows are indexed by ICUSTAY_ID and the columns come from diagnosis_labels. The values
    are binary.

    The paper by Harutyunyan et al. did not reveal how diagnosis_labels were selected. In the methods subsection
    "Acute care phenotype classification," the authors state that they used 25 categories from the HCUP Clinical
    Classification Software (2015 version for ICD-9 codes) for phenotype prediction and selected ICD-9 codes that
    mapped to the 25 categories. However, based on information in HCUP's own mapping files, there are far more ICD-9 
    codes that map to the 25 categories than what is in the diagnosis_labels list.
    """
    diagnoses_copy = diagnoses.copy()
    diagnoses_copy['VALUE'] = 1
    labels = diagnoses_copy[['ICUSTAY_ID', 'ICD_CODE', 'VALUE']].drop_duplicates().set_index('ICUSTAY_ID')
    labels = labels.pivot(columns='ICD_CODE', values='VALUE').fillna(0).astype(int)
    labels = labels.reindex(columns=diagnosis_labels, fill_value=0)
    labels = labels[diagnosis_labels]
    return labels.rename({d: 'Diagnosis ' + d for d in diagnosis_labels}, axis=1)


def make_phenotype_label_matrix(
    phenotypes: DataFrame, 
    icd9_definitions: dict,
    icd10_definitions: dict,
    stays: Optional[DataFrame]=None
) -> DataFrame:

    def _update_def_map_and_categories(definitions, def_map, benchmark_categories):
        for category in definitions:
            if not definitions[category]['use_in_benchmark']:
                continue
            else:
                benchmark_categories.add(category)
                for code in definitions[category]['codes']:
                    if code in def_map:
                        if category not in def_map[code]:
                            def_map[code].append(category)
                    else:
                        def_map[code] = [category]
        return def_map, benchmark_categories

    # Create a dictionary of ICD codes that are members of benchmark HCUP categories and list those categories
    benchmark_categories = set()
    icd9_def_map = dict()
    icd10_def_map = dict()

    icd9_def_map, benchmark_categories = _update_def_map_and_categories(
        icd9_definitions, icd9_def_map, benchmark_categories
    )
    icd10_def_map, benchmark_categories = _update_def_map_and_categories(
        icd10_definitions, icd10_def_map, benchmark_categories
    )

    benchmark_categories = sorted(list(benchmark_categories))

    # Initialize columns for each HCUP category used in the benchmark
    phenotypes = phenotypes.copy(deep=True)
    for category in benchmark_categories:
        phenotypes[category] = 0
    # Assign 1 to categories that the ICD codes belong to
    for idx, row in phenotypes.iterrows():
        def_map = icd9_def_map if int(row['ICD_VERSION']) == 9 else icd10_def_map
        if row['ICD_CODE'] in def_map:
            for category in def_map[row['ICD_CODE']]:
                phenotypes.loc[idx, category] = 1
    # Create phenotype label matrix
    phenotype_labels = phenotypes.groupby('ICUSTAY_ID')[benchmark_categories].max()
    if stays is not None:
        phenotype_labels = phenotype_labels.reindex(stays.ICUSTAY_ID.sort_values())
    return phenotype_labels.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)


def add_hcup_groups(diagnoses, icd_version, definitions):
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            if code in def_map:
                # Add the category to the list of categories for this code
                def_map[code][0].append(dx)
                # If the code has already been flagged for use in benchmark, keep it that way.
                if not def_map[code][1]:
                    def_map[code][1] = definitions[dx]['use_in_benchmark']
            else:
                def_map[code] = [[dx], definitions[dx]['use_in_benchmark']]
    if icd_version == 9:
        sel_rows = diagnoses['ICD_VERSION'].astype(int) == 9
        new_col_name = 'HCUP_CCS_2015'
    else:
        sel_rows = diagnoses['ICD_VERSION'].astype(int) == 10
        new_col_name = 'HCUP_CCSR_2024'
    # If there are multiple HCUP categories for a single ICD code, concatenate them into a "; "-delimited string
    # This does not apply to HCUP CCS 2015 categories as ICD 9 codes map to a single category.
    diagnoses.loc[sel_rows, new_col_name] = diagnoses.loc[sel_rows, 'ICD_CODE'].apply(
        lambda c: '; '.join(def_map[c][0]) if c in def_map else None
    )
    diagnoses.loc[sel_rows, 'USE_IN_BENCHMARK'] = diagnoses.loc[sel_rows, 'ICD_CODE'].apply(
        lambda c: int(def_map[c][1]) if c in def_map else None
    )
    return diagnoses


def read_itemid_to_variable_map(file_path: str, variable_column: str = 'VARIABLE') -> DataFrame:
    """
    Read a CSV file containing a mapping from item IDs to the names of variables that will be extracted from the MIMIC-IV data tables for use as features in predictive models. At minimum, the CSV file must contain the following columns:

    - ITEMID: The item ID number from the MIMIC-IV database. These can be found in MIMIC-IV tables whose filenames start with "d_".
    - VARIABLE: The final name that you want to use for the variable in your model. These need not map 1:1 with the ITEMID. For example, you may want to assign multiple ITEMIDs to the same variable name, since similar or identical features in the MIMIC-IV database often fall under multiple different item IDs.
    - LABEL: The variable name that is used in the MIMIC-IV database. Each item ID has a single (but not necessarily unique) label.
    - UNITNAME: The unit of measurement for the variable, if applicable. This may be used for unit conversion.

    Args:
    
        file_path (str): Path to the CSV file containing a mapping of item IDs to variable names.
        variable_column (str): Name of the column containing the variable names.

    Returns:
    
        DataFrame: A DataFrame with ITEMID as index and VARIABLE and MIMIC_LABEL as columns.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    var_map = dataframe_from_csv(file_path, index_col='ITEMID')
    incl = pd.notna(var_map[variable_column])
    var_map = var_map.loc[incl, :]

    return var_map


def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, how='inner', left_on='ITEMID', right_index=True)
