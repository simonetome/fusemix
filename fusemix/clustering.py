# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:58:29 2025

@author: simon

functions to calculate clustering assignments

Functions:
    
    compute_kpod: perform kpod algorithm

"""

import snf

from sklearn.cluster import spectral_clustering, KMeans
from sklearn.impute import KNNImputer

from gower import gower_matrix
from kPOD import k_pod

import numpy as np
from numpy.typing import ArrayLike 

import pandas as pd


def compute_kpod(
    incomplete_data,
    num_clusters,
    seed
    ):
    """
    Function to compute kPod algorithm

    Args:
        incomplete_data (pandas.DataFrame): dataset with possible missing values
        num_clusters (integer): number of clusters
        seed (integer): random number seed

    Returns:
        ArrayLike: list of clustering assignments 
    """
    return k_pod(incomplete_data, num_clusters, random_state=seed)[0]



def compute_spectral_si_knn(
    incomplete_data,
    cat_mask,
    num_clusters,
    seed
):
    """
    Compute spectral clustering on a single imputation
    Algorithm: 
     - impute using KNN with gower distance metric 
     - perform spectral equally as CCA
    """
    imputer = KNNImputer(
        missing_values=np.nan, 
        n_neighbors=5, 
        metric="nan_euclidean",
    )
    single_imputed_data = imputer.fit_transform(incomplete_data)
    sc_complete = compute_spectral_complete(
        complete_data=single_imputed_data,
        cat_mask=cat_mask,
        num_clusters=num_clusters,
        seed=seed
    )
    return sc_complete

def compute_spectral_si_mi(
    multiple_imputed_data,
    cat_mask,
    num_clusters,
    seed
):
    """
    Compute spectral clustering on a single imputation
    Algorithm: 
     - impute using mean of multiple imputations
     - perform spectral equally as CCA
    """
    view = multiple_imputed_data[0]

    single_imputed_data = pd.DataFrame(
        np.mean(np.array(multiple_imputed_data), axis = 0), 
        columns=view.columns, 
        index=view.index
        )
    
    sc_complete = compute_spectral_complete(
        complete_data=single_imputed_data,
        cat_mask=cat_mask,
        num_clusters=num_clusters,
        seed=seed
    )
    return sc_complete

def compute_kmeans_si_knn(
    incomplete_data,
    num_clusters,
    seed
):
    """
    Compute spectral clustering on a single imputation
    Algorithm: 
     - impute using KNN with gower distance metric 
     - perform kmeans equally as CCA
    """
    imputer = KNNImputer(
        missing_values=np.nan, 
        n_neighbors=5, 
        metric="nan_euclidean",
    )
    single_imputed_data = imputer.fit_transform(incomplete_data)
    sc_complete = compute_kmeans_complete(
        complete_data=single_imputed_data,
        num_clusters=num_clusters,
        seed=seed
    )
    return sc_complete

def compute_kmeans_si_mi(
    multiple_imputed_data,
    num_clusters,
    seed
):
    """
    Compute spectral clustering on a single imputation
    Algorithm: 
     - impute using KNN with gower distance metric 
     - perform kmeans equally as CCA
    """
    view = multiple_imputed_data[0]

    single_imputed_data = pd.DataFrame(
        np.mean(np.array(multiple_imputed_data), axis = 0), 
        columns=view.columns, 
        index=view.index
        )
    km_complete = compute_kmeans_complete(
        complete_data=single_imputed_data,
        num_clusters=num_clusters,
        seed=seed
    )
    return km_complete


def compute_fusemix(
    multiple_imputed_data,
    cat_mask,
    num_clusters,
    nn_snf,
    seed
    ):
    """
    Experimental
    """
    affinities = [snf.compute.make_affinity(d,metric = "gower",K=nn_snf,cat_features = cat_mask) for d in multiple_imputed_data]
    fused_network = snf.snf(affinities, K=nn_snf)
    fusemix_labels = spectral_clustering(affinity=fused_network, n_clusters=num_clusters, random_state=seed)
    return fusemix_labels


def compute_spectral_complete(
    complete_data,
    cat_mask,
    num_clusters,
    seed,
    ) -> ArrayLike:
    """
    Compute spectral clustering on complete data.
    In this case the heat kernel of affinity matrix is used
    """
    gower_dist_complete = gower_matrix(complete_data,cat_features=cat_mask)
    spectral_labels = spectral_clustering(
        affinity=__heat_kernel_affinity(gower_dist_complete),
        n_clusters=num_clusters,
        random_state=seed,
        )
    return spectral_labels


def compute_kmeans_complete(
    complete_data,
    num_clusters,
    seed,
    ) -> ArrayLike:
    km = KMeans(n_clusters=num_clusters,random_state=seed)
    km_labels = km.fit(complete_data).labels_
    return km_labels


def __heat_kernel_affinity(distance_matrix, sigma=None):
    """
    Compute heat kernel from affinity matrix
    """
    # If sigma is not provided, set it to median distance (common heuristic)
    if sigma is None:
        # Avoid diagonal zeros
        nonzero_dists = distance_matrix[distance_matrix > 0]
        sigma = np.median(nonzero_dists)

    # heat kernel
    K = np.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))
    return K









