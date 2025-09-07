"""
This script defines function useful to fetch/load data for tests/productions
"""
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo


def load_uciml(id: int = None, name: str = None, na_drop: bool = False):
    """
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




