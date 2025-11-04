"""
This file contains wrappers to call models from the MixtureMissing CRAN R package

Docs:
https://cran.r-project.org/web/packages/MixtureMissing/refman/MixtureMissing.html#MCNM

2 models:

- Multivariate Contaminated Normal Mixture (MCNM) - Tong and Tortora, 2022 (https://doi.org/10.1007%2Fs11634-021-00476-1)
- Multivariate Generalized Hyperbolic Mixture (MGHM) - Wei et. al 2019 (https://doi.org/10.1016%2Fj.csda.2018.08.016)

"""

from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, NULL
from rpy2.robjects.conversion import localconverter

import pandas as pd 
import numpy as np

MixtureMissing = importr('MixtureMissing')
base = importr("base")


def run_mcnm(
    df_py, 
    G=2, 
    seed=None,
    init_method="kmeans", 
    epsilon_start=1e-3, 
    max_iter=20, 
    max_epsilon=1.0):
    """
    Run MCNM on a pandas DataFrame with adaptive epsilon until convergence.
    
    Parameters:
        df_py: pandas.DataFrame
        G: int, number of clusters
        init_method: str, initialization method
        epsilon_start: float, starting epsilon
        max_iter: int, maximum iterations per call
        max_epsilon: float, maximum epsilon to try
    
    Returns:
        res_py: Python dict of the MCNM result
    """
    eps = epsilon_start
    res_r = NULL
    base.set_seed(seed)

    with localconverter(pandas2ri.converter):
        df_r = pandas2ri.py2rpy(df_py)
    
    mcnm = r["MCNM"]
    
    while res_r == NULL and eps <= max_epsilon:
        try:
            res_r = mcnm(
                X=df_r,
                G=r['as.integer'](G),
                init_method=init_method,
                epsilon=r['as.numeric'](eps),
                max_iter=r['as.integer'](max_iter),
                progress=False
            )
            
        except Exception as e:
            res_r = None
        
        if res_r == NULL:
            eps *= 10
    
    if res_r == NULL:
        raise RuntimeError("MCNM did not converge even after increasing epsilon.")
    
    clusters_r = res_r.rx2('clusters')
    with localconverter(pandas2ri.converter):
        clusters_py = pd.Series(pandas2ri.rpy2py(clusters_r))
    
    return np.array(clusters_py)




def run_mghm(
    df_py, 
    G=2, 
    seed=None,
    init_method="kmeans", 
    epsilon_start=1e-3, 
    max_iter=20, 
    max_epsilon=1.0):
    """
    Run MGHM on a pandas DataFrame with adaptive epsilon until convergence.
    
    Parameters:
        df_py: pandas.DataFrame
        G: int, number of clusters
        init_method: str, initialization method
        epsilon_start: float, starting epsilon
        max_iter: int, maximum iterations per call
        max_epsilon: float, maximum epsilon to try
    
    Returns:
        res_py: Python dict of the MCNM result
    """
    eps = epsilon_start
    res_r = NULL
    base.set_seed(seed)

    with localconverter(pandas2ri.converter):
        df_r = pandas2ri.py2rpy(df_py)
    
    mghm = r["MGHM"]
    
    while res_r == NULL and eps <= max_epsilon:
        try:
            res_r = mghm(
                X=df_r,
                G=r['as.integer'](G),
                init_method=init_method,
                epsilon=r['as.numeric'](eps),
                max_iter=r['as.integer'](max_iter),
                progress=False,
                outlier_cutoff=1
            )
            
        except Exception as e:
            res_r = None
        
        if res_r == NULL:
            eps *= 10
    
    if res_r == NULL:
        raise RuntimeError("MCNM did not converge even after increasing epsilon.")
    
    clusters_r = res_r.rx2('clusters')
    with localconverter(pandas2ri.converter):
        clusters_py = pd.Series(pandas2ri.rpy2py(clusters_r))
    
    return np.array(clusters_py)