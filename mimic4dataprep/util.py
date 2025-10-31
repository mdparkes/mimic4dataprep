from __future__ import absolute_import
from __future__ import print_function

import os
import pandas as pd


def dataframe_from_csv(path, header=0, index_col=False):
    df=pd.read_csv(path, header=header, index_col=index_col)
    df.columns = df.columns.str.upper()
    df.rename(columns={"STAY_ID": "ICUSTAY_ID", "CAREGIVER_ID": "CGID", "RACE": "ETHNICITY"}, inplace = True)
    return df


def get_resources_dir_path() -> str:
    """
    Get the path to the 'resources' directory.

    Returns:
        str
            Path to the 'resources' directory.
    """
    return os.path.join(os.path.dirname(__file__), 'resources')
