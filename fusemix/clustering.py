# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:58:29 2025

@author: simon

functions to calculate clustering assignments

"""

import snf

from numpy.typing import ArrayLike 
from sklearn.cluster import spectral_clustering, KMeans
from gower import gower_matrix
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, completeness_score, v_measure_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from kPOD import k_pod

import numpy as np

def compute_fusemix(multiple_imputed_data,
                    cat_mask,
                    num_clusters,
                    nn_snf,
                    seed):
    affinities = [snf.compute.make_affinity(d,metric = "gower",K=nn_snf,cat_features = cat_mask) for d in multiple_imputed_data]
    fused_network = snf.snf(affinities, K=nn_snf)
    fusemix_labels = spectral_clustering(affinity=fused_network, n_clusters=num_clusters, random_state=seed)
    return fusemix_labels


def compute_MICA(multiple_imputed_data,
                 num_clusters,
                 seed):
    
    # compute clustering for each view
    results = []
    for view in multiple_imputed_data: # view is a imputed dataset 
        kmeans = KMeans(init="k-means++", 
                        n_clusters=num_clusters,
                        n_init=1, 
                        random_state=seed)
        results += [kmeans.fit(view).cluster_centers_]

    # clustering centroids 
    kmean_centroids = KMeans(init="k-means++", 
                            n_clusters=num_clusters,
                            n_init="auto", 
                            random_state=seed)
    kmean_centroids = kmean_centroids.fit(np.vstack(results))

    predictions = [kmean_centroids.predict(X=view) for view in multiple_imputed_data]

    majority_votes = np.array([
        np.bincount(col).argmax() for col in np.vstack(predictions).T
    ])

    return majority_votes


def compute_spectral(complete_data,
                     cat_mask,
                     num_clusters,
                     seed) -> ArrayLike:
    gower_dist_complete = gower_matrix(complete_data,
                                        cat_features=cat_mask)
    spectral_labels = spectral_clustering(affinity=(1 - gower_dist_complete),
                                                    n_clusters=num_clusters,
                                                    random_state=seed)
    return spectral_labels

def compute_kmeans(complete_data,
                     num_clusters,
                     seed) -> ArrayLike:
    km = KMeans(n_clusters=num_clusters,random_state=seed)
    km_labels = km.fit(complete_data).labels_
    return km_labels




def compute_kpod(incomplete_data,
                 num_clusters,
                 seed):
    
    return k_pod(incomplete_data, num_clusters, random_state=seed)


def external_metrics(true_labels,predicted_labels):
    """
    function to compute external metrics related to clustering
    external metrics consider that there is a set of true labels, which in our case 
    can be either the CCA labels or the true classes of the samples
    """
    ari = adjusted_rand_score(true_labels,predicted_labels)
    ami = adjusted_mutual_info_score(true_labels,predicted_labels)
    cs = completeness_score(true_labels, predicted_labels)
    vm = v_measure_score(true_labels, predicted_labels)
    return {'ari':ari,'ami':ami,'vm': vm, 'cs':cs}    


def internal_metrics(predicted_labels, gower_dist, complete_data):
    """
    internal validation metrics are tailored to clustering and can evaluate 
    different qualities of a clustering results, e.g., separability, compactness, etc.

    there are no true labels

    sil -> higher means better
    ch -> higher means better
    db -> lower means better

    """
    sil_score = silhouette_score(X=gower_dist,metric="precomputed",labels=predicted_labels)
    dbouldin_score = davies_bouldin_score(X=complete_data, labels=predicted_labels)
    charabasz_score = calinski_harabasz_score(X=complete_data, labels=predicted_labels)

    return {'sh':sil_score,
            'ch':charabasz_score,
            'db':dbouldin_score}







