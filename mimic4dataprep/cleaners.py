import pandas as pd
import numpy as np
import warnings

from datacleaner.datacleaner import DataCleaner
from typing import Union
from numpy.typing import ArrayLike


"""
The guiding principle for data cleaning is to keep it as simple as possible. It is motivated by the belief that a model's likelihood of being used in a clinical setting is inversely related to the effort required to get it up and running, and by the reality that medical data usually contain incorrectly entered values and outliers. In the spirit of simplicity and training a model that is robust against noisy data, the following set of simple, sweeping transformations is applied to the the selected features from the MIMIC-IV data:

Apply to all numeric fields in sequence:
- remove_non_numeric_strings
- cast_as_float
- remove_negative_values

If numeric and unit is %:
- clean_percent

Apply to all text fields:
- empty_string_to_nan


Apply to specific numeric fields:
- convert_inch_to_cm
- convert_lb_to_kg
- convert_f_to_c

Apply to specific text fields:
- convert_capillary_refill_rate
- convert_gcs
"""


def convert_units(value: Union[float, ArrayLike], from_unit: str, to_unit: str) -> Union[float, ArrayLike]:
    """Convert a value from one unit to another.
    
    If no method is available for converting the units the original value is returned with a warning.
    """

    # Temperature
    if from_unit == 'F':
        if to_unit == 'C':
            return (value - 32) * 5.0 / 9.0

    if from_unit == 'C':
        if to_unit == 'F':
            return value * 9.0 / 5.0 + 32

    # Weight
    if from_unit == 'lb':
        if to_unit == 'kg':
            return value * 0.453592
        if to_unit == 'g':
            return value * 453.592
        if to_unit == 'oz':
            return value * 16.0
    
    if from_unit == 'oz':
        if to_unit == 'kg':
            return value * 0.0283495
        if to_unit == 'g':
            return value * 28.3495
        if to_unit == 'lb':
            return value / 16.0
    
    if from_unit == 'kg':
        if to_unit == 'lb':
            return value / 0.453592
        if to_unit == 'oz':
            return value * 35.274
        if to_unit == 'g':
            return value * 1000.0

    if from_unit == 'g':
        if to_unit == 'lb':
            return value / 453.592
        if to_unit == 'oz':
            return value / 28.3495
        if to_unit == 'kg':
            return value / 1000.0    
    
    if from_unit == 'in':
        if to_unit == 'cm':
            return value * 2.54
        
    if from_unit == 'cm':
        if to_unit == 'in':
            return value / 2.54

    warnings.warn(f"No method available to convert from {from_unit} to {to_unit}. Returning original value.")

    return value


def empty_string_to_nan(x: pd.Series) -> pd.Series:
    # Replace empty strings with NaN
    return x.where(x != '', other=np.nan)


def remove_nonnumeric_strings(x: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(x):
        # `regexp` will match strings that can be cast as floats
        regexp = r'^(?:\d+(?:\.\d*)?|\.\d+)$'
        # Keep strings that can be cast as float, replace others with NaN
        x = x.where(x.str.match(regexp), other=np.nan)
    return x


def cast_as_float(x: pd.Series) -> pd.Series:
    return x.astype(float)


def remove_negative_values(x: pd.Series) -> pd.Series:
    return x.where(x >= 0, other=np.nan)


def clean_percent(x: pd.Series) -> pd.Series:
    # If value is in [0, 1], assume that it needs to be converted to a percentage
    sel = x.between(0, 1, inclusive='both')
    x.loc[sel] = x.loc[sel] * 100
    # Remove anything outside of [0, 100]
    x = x.where(x.between(0, 100, inclusive='both'), other=np.nan)
    return x


def convert_capillary_refill_rate(x: pd.Series) -> pd.Series:
    mapping = {
        'Normal <3 Seconds': 0,
        'Abnormal >3 Seconds': 1
    }
    x = x.where(x.isin(mapping.keys()), other=np.nan)
    # Replace description with value
    with pd.option_context('future.no_silent_downcasting', True):
        x = x.replace(mapping).astype(float)
    return x


def convert_gcs(x: pd.Series) -> pd.Series:
    mapping = {
        # Eye opening
        'No eye opening': 1,  # Not encountered in MIMIC-IV data
        'To Pain': 2,
        'To Speech': 3,
        'Spontaneously': 4,
        # Verbal
        'No Response-ETT': np.nan,  # Endotracheal tube -- Can't assess
        'No Response': 1,
        'Incomprehensible sounds': 2,
        'Inappropriate Words': 3,
        'Confused': 4,
        'Oriented': 5,
        # Motor
        'No response': 1,
        'Abnormal extension': 2,
        'Abnormal Flexion': 3,
        'Flex-withdraws': 4,
        'Localizes to Pain': 5,
        'Obeys Commands': 6
    }
    # Replace empty strings with NaN (mapping covers every possible value in MIMIC-IV)
    x = x.where(x.isin(mapping.keys()), other=np.nan)
    # Replace description with value
    with pd.option_context('future.no_silent_downcasting', True):
        x = x.replace(mapping).astype(float)
    return x


def convert_inch_to_cm(x: pd.Series) -> pd.Series:
    return convert_units(x, from_unit='in', to_unit='cm')


def convert_lb_to_kg(x: pd.Series) -> pd.Series:
    return convert_units(x, from_unit='lb', to_unit='kg')


def convert_f_to_c(x: pd.Series) -> pd.Series:
    return convert_units(x, from_unit='F', to_unit='C')


def clean_events(events: pd.DataFrame, var_map: pd.DataFrame) -> pd.DataFrame:

    sel_cols = ['VARIABLE', 'PARAM_TYPE', 'ITEMID', 'UNITNAME']
    grouped_vars = var_map.reset_index().loc[:, sel_cols].groupby(['VARIABLE', 'PARAM_TYPE'])
    
    dc = DataCleaner()  # Initialize a DataCleaner instance
    dc.add_cleaner(remove_nonnumeric_strings)
    dc.add_cleaner(cast_as_float)
    dc.add_cleaner(remove_negative_values)
    dc.add_cleaner(empty_string_to_nan)
    dc.add_cleaner(clean_percent)
    dc.add_cleaner(convert_capillary_refill_rate)
    dc.add_cleaner(convert_gcs)
    dc.add_cleaner(convert_inch_to_cm)
    dc.add_cleaner(convert_lb_to_kg)
    dc.add_cleaner(convert_f_to_c)


    # Associate variables with cleaning pipelines to be executed by DataCleaner
    for (name, type), info in grouped_vars:

        if type == 'Numeric':
            for tup in info.itertuples(index=False):
                itemid = tup.ITEMID
                unitname = tup.UNITNAME
                # Cleaners applied to all numeric variables
                dc.add_variable(
                    variable_name=itemid,
                    cleaners=[
                        'remove_nonnumeric_strings',
                        'cast_as_float',
                        'remove_negative_values'
                    ]
                )
                # Additional cleaners applied to percentages
                if unitname == '%':
                    dc.update_variable(
                        variable_name=itemid,
                        cleaners=[
                            'clean_percent'
                        ],
                        append_cleaners=True  # Append to previously added cleaners
                    )
        
        elif type == 'Text':
            for tup in info.itertuples(index=False):
                itemid = tup.ITEMID
                unitname = tup.UNITNAME
                # Cleaners applied to all text variables
                dc.add_variable(
                    variable_name=itemid,
                    cleaners=[
                        'empty_string_to_nan'
                    ]
                )
    
    # Apply specific cleaners to specific variables

    # Capillary refill rate
    for itemid in [223951, 224308]:
        dc.update_variable(
            variable_name=itemid,
            cleaners=[
                'convert_capillary_refill_rate'
            ],
            append_cleaners=True  # Append to previously added cleaners
        )

    # GCS
    for itemid in [220739, 223901, 223900]:
        dc.update_variable(
            variable_name=itemid,
            cleaners=[
                'convert_gcs'
            ],
            append_cleaners=True  # Append to previously added cleaners
        )
    
    # Height
    dc.update_variable(
        variable_name=226707,
        cleaners=[
            'convert_inch_to_cm'
        ],
        append_cleaners=True  # Append to previously added cleaners
    )

    # Weight
    dc.update_variable(
        variable_name=226531,
        cleaners=[
            'convert_lb_to_kg'
        ],
        append_cleaners=True  # Append to previously added cleaners
    )

    # Temperature
    dc.update_variable(
        variable_name=223761,
        cleaners=[
            'convert_f_to_c'
        ],
        append_cleaners=True  # Append to previously added cleaners
    )

    # Apply the cleaning pipelines
    cleaned_events = dc(events, value_column='VALUE', variable_column='ITEMID')

    return cleaned_events
