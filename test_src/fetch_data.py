"""
Script to fetch data from UCI repository
"""

import sys
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
import pickle
import numpy as np


def load_uciml(id: int = None, name: str = None, na_drop: bool = False):
    """
    Function to load data from the UCI Machine learning repository
    Args:
        id:
        name:
    Returns:
    """

    if id is None:
        dataset = fetch_ucirepo(name=name)
    else:
        dataset = fetch_ucirepo(id=id)

    # access data
    X = dataset.data.features
    if na_drop:
        X = X[pd.isna(X).sum()[~np.bool(pd.isna(X).sum().values)].index]

    # Remove duplicate columns (happened in ID 174)
    X = X.loc[:, ~X.columns.duplicated()]
    
    dataset.variables = dataset.variables.loc[~dataset.variables['name'].duplicated(),:]
    dataset.variables = dataset.variables.reset_index(drop=True)

    # sanitize names
    X.columns = X.columns.str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
    dataset.variables['name'] = dataset.variables['name'].str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)

    y = dataset.data.targets

    cat_variables = dataset.variables.loc[(dataset.variables['type'] == 'Categorical') | (dataset.variables['type'] == 'Binary'), 'name']
    num_variables = dataset.variables.loc[(dataset.variables['type'] != 'Categorical') & (dataset.variables['type'] != 'Binary'), 'name']

    features = dataset.data.features.columns

    cat_variables = cat_variables[cat_variables.isin(features)]
    num_variables = num_variables[num_variables.isin(features)]

    obj_cols = X.select_dtypes(include='object').columns

    X = X.copy()
    X.loc[:,obj_cols] = X.loc[:,obj_cols].apply(lambda col: col.astype('category').cat.codes.astype(float))
    X.loc[:, num_variables] = X.loc[:, num_variables].astype('float64')

    cat_variables_in_X = cat_variables[cat_variables.isin(X.columns)]
    cat_mask = X.columns.isin(cat_variables_in_X)

    complete_rows_mask = ~(X.isnull().sum(axis = 1) > 0)
    
    X_complete = X.loc[complete_rows_mask,:].copy()
    X_complete = X_complete.apply(pd.to_numeric, errors="coerce")
    X_complete = X_complete.astype('float64')

    y_complete = y[complete_rows_mask].copy()

    res = {}
    res['X'] = X.copy()
    res['y'] = y
    res['cat_variables'] = cat_variables
    res['cat_variables_in_X'] = cat_variables_in_X
    res['cat_mask'] = cat_mask
    res['id'] = id

    res['X_complete'] = X_complete.copy()
    res['y_complete'] = y_complete

    res['num_classes'] = y_complete.value_counts().shape[0]


    return res


notebook = False
BASE_PATH = os.path.abspath(os.path.dirname(sys.argv[0]))

if notebook:
    BASE_PATH = os.getcwd()
    
FETCHED_PATH = os.path.normpath(os.path.join(BASE_PATH, "../test_data/fetched"))

dataset_ids_path = os.path.join(FETCHED_PATH,"dataset_ids.csv")
dataset_ids = pd.read_csv(dataset_ids_path, sep = " ")
print(dataset_ids.head(3))

datasets = dict.fromkeys(dataset_ids['Id'])


for id in dataset_ids['Id']:
    print(f"Collecting dataset {id}")
    datasets[id] = load_uciml(id=id)


for d in datasets:
    print(os.path.join(FETCHED_PATH,"dataset_"+str(d)+".pkl"))
    with open(os.path.join(FETCHED_PATH,"dataset_"+str(d)+".pkl"),"wb") as out:
        pickle.dump(datasets[d],out)