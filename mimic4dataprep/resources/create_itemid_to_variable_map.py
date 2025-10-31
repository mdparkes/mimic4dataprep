import argparse
import numpy as np
import pandas as pd

from mimic4benchmark.mimic4csv import get_table_file_path
from tqdm import tqdm
import os
import re

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create itemid to variable map.')
    parser.add_argument('mimic4_path', type=str, help='Directory containing MIMIC-IV CSV files.')
    args, _ = parser.parse_known_args()

    # Read the d_items table
    d_items_path = get_table_file_path(args.mimic4_path, 'd_items')
    d_items = pd.read_csv(d_items_path, low_memory=False)
    d_items.columns = d_items.columns.str.upper()
    d_items.set_index('ITEMID', inplace=True)

    # Read the d_labevents table
    d_labitems_path = get_table_file_path(args.mimic4_path, 'd_labitems')
    d_labitems = pd.read_csv(d_labitems_path, low_memory=False)
    d_labitems.columns = d_labitems.columns.str.upper()
    d_labitems.set_index('ITEMID', inplace=True)

    # Add a LINKSTO column to d_labitems
    # Initialize LINKSTO column in d_labitems
    d_labitems['LINKSTO'] = pd.Series(index=d_labitems.index, dtype='object')
    d_labitems['UNITNAME'] = pd.Series(index=d_labitems.index, dtype='object')

    # Iterate through CSV files in the hosp directory
    hosp_path = os.path.join(args.mimic4_path, 'hosp')

    for root, dirs, files in os.walk(args.mimic4_path):
        for file in files:
            if file.endswith('.csv') and not file.startswith('d_'):
                table_name = re.sub(r'\.csv$', '', file)
                file_path = os.path.join(root, file)
                try:
                    # Read only the itemid, valueuom columns if they exist
                    columns = pd.read_csv(file_path, nrows=0).columns
                    columns = columns[columns.str.contains('itemid|valueuom')]
                    has_valueuom_col = 'valueuom' in columns
                    if 'itemid' in columns:
                        # Only read necessary columns
                        df = pd.read_csv(file_path, usecols=columns, dtype={'itemid': 'int64'})

                        # Filter for only those itemids that exist in d_labitems
                        incl = df['itemid'].isin(d_labitems.index)
                        valid_itemids = df.loc[incl, 'itemid']
                        if not valid_itemids.empty:
                            d_labitems.loc[valid_itemids, 'LINKSTO'] = table_name
                        
                        # Update UNITNAME if valueuom exists in the table
                        if has_valueuom_col:
                            # Group by itemid and take first valueuom
                            unit_map = df.dropna(subset=['valueuom']).drop_duplicates('itemid')
                            unit_map = unit_map.set_index('itemid')['valueuom']
                            # Update only those rows that match and exist in d_labitems
                            valid_units = unit_map.index.intersection(d_labitems.index)
                            if not valid_units.empty:
                                d_labitems.loc[valid_units, 'UNITNAME'] = unit_map[valid_units]
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Merge the two tables on ITEMID and clean up columns
    merged = pd.merge(d_items, d_labitems, on='ITEMID', how='outer')
    merged['LABEL'] = merged['LABEL_x'].combine_first(merged['LABEL_y'])
    merged['CATEGORY'] = merged['CATEGORY_x'].combine_first(merged['CATEGORY_y'])
    merged['LINKSTO'] = merged['LINKSTO_x'].combine_first(merged['LINKSTO_y'])
    merged['UNITNAME'] = merged['UNITNAME_x'].combine_first(merged['UNITNAME_y'])
    columns_to_drop = [col for col in merged.columns if re.search(r'_[xy]$', col)]
    merged.drop(columns=columns_to_drop, inplace=True)
    merged['COUNT'] = 0  # Count of itemid occurrences in tables of records in 'LINKSTO'
    columns = ['LABEL', 'ABBREVIATION', 'UNITNAME', 'PARAM_TYPE', 'LOWNORMALVALUE', 'HIGHNORMALVALUE',
               'LINKSTO', 'CATEGORY', 'FLUID', 'COUNT']
    merged = merged.loc[:, columns]  # Specify column order

    # Count the number of times each ITEMID appears in the table of records in 'LINKSTO'
    tables = merged['LINKSTO'].unique()  # Tables to which the ITEMID links
    for table in tables:
        if table in ['d_items', 'd_labitems', np.nan]:
            continue
        print(f'Counting occurrences of item IDs in {table} table')
        fp = get_table_file_path(args.mimic4_path, table)
        # Read only the itemid column
        reader = pd.read_csv(
            fp, usecols=['itemid'], dtype={'itemid': 'int64'}, iterator=True, chunksize=1e6, encoding='windows-1252'
        )
        for chunk in tqdm(reader, desc=f'Processing {table}'):
            # Count occurrences of each itemid in this chunk
            counts = chunk['itemid'].value_counts()
            # Update the counts in the merged dataframe
            for itemid, count in counts.items():
                if itemid in merged.index:
                    merged.at[itemid, 'COUNT'] += count
    
    # Add new features
    df = pd.DataFrame({
        'LABEL': 'Discharge Summary',
        'PARAM_TYPE': 'Text',
        'LINKSTO': 'discharge',
        'COUNT': 0
        }, index=pd.Index([1000000], name='ITEMID')
    )
    merged = pd.concat([merged, df], axis=0, join='outer', ignore_index=False)

    # Write to disk
    merged.to_csv('mimic4benchmark/resources/itemid_to_variable_map_mimic4.csv', index=True)
