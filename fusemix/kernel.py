# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:53:17 2025

All the implementation derives from the thesis:
    'kernel methods with mixed data types and their application'

@author: simon
"""
import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd


dataset = fetch_ucirepo(name = 'Heart Disease')
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = dataset.data.features
y = dataset.data.targets


dataset.variables['type'] 
cat_variables = dataset.variables.loc[dataset.variables['type'] == 'Categorical', 'name'].values



# given an array of samples for a single feature, computes the NxN square 
# boolean matrix where the element i,j = 1 if x_i == x_j
def calc_eq_matrix(arr: np.ndarray):
    return arr[:, None] == arr[None, :]


# return the categorical gram matrix
def calculate_categorical_kernel(dataset: pd.DataFrame,
                                 cat_variables: list[str],
                                 alpha: float = 1,
                                 gamma: float = 1e-3):
    # Precompute frequencies for each categorical variable
    freqs = {}
    for l in cat_variables:
        freqs[l] = dataset.loc[:,l].value_counts()/X.shape[0] 

    # Precumpute kernel values in case of match
    precomp_cat_kernel_values = {}
    for cat_var, f_values in freqs.items():
        precomp_cat_kernel_values[cat_var] = (1-f_values**alpha)**(1/alpha)
    
    # each categorical feature gets its own equality matrix
    eq_matrixes = {
        cat_var: calc_eq_matrix(X[cat_var].to_numpy())
        for cat_var in cat_variables
    }

    # Compute final kernel values
    results = {}
    for col in cat_variables:
        kernel_vals = X[col].map(precomp_cat_kernel_values[col]).to_numpy()
        results[col] = eq_matrixes[col] * kernel_vals[:, None]

    additive_kernel = np.add.reduce(list(results.values()))/len(cat_variables)
    
    # See notes to see how gamma affects kernel values
    aggregated_kernel = (np.exp(gamma*additive_kernel)-1)/(np.exp(gamma)-1)

    return aggregated_kernel



















