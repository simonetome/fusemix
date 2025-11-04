"""
Algorithm MIGEClust: our proposal 

for each view:

    - p' = p_min + alfa[p_max - p_min] 
        being alfa random between 0 and 1
        p_min = p*0.75
        p_max = p*0.85 Yu et. al 

    - sample p' features from the view and create M' projected datasets
    
    for each projected dataset 
        - Compute Gower similarity matrix 
        - Compute sparsified adjacency matrix using KNN 
        -    
"""

from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix

from gower import gower_matrix

import numpy as np

def mige(
    multiple_imputed_data,
    n_clusters,
    cat_mask = None,
    seed = None,
    p_min = 0.75,
    p_max = 0.85,
    num_projections = 5,
    k_nn = 10,
    co_threshold = 0.5
    ):
    """
    MIGEClust function

    cat_mask is required for mixed type data with gower distance
    returns label of clustering as np.array
    """

    # random seed generator
    rng = np.random.default_rng(seed)

    n_features = multiple_imputed_data[0].shape[1]
    p_min_ = p_min*n_features
    p_max_ = p_max*n_features
    
    # Generate all projections for all data
    if num_projections > 0:
        all_projections  = []
        for view in multiple_imputed_data:
            for i in range(num_projections):
                all_projections.append(__generate_projection(
                    data=view, 
                    cat_mask=cat_mask,
                    n_features=n_features,
                    rng=rng,
                    p_min_=p_min_,
                    p_max_=p_max_
                    ))
    else:
        # if i don't want to project, I simply use the multiple imputed data
        all_projections = multiple_imputed_data

    # compute sparse similarity graphs for each projection 
    sparse_graphs = [__compute_sparse_similarity(prj_view,prj_mask,k_nn) for (prj_view, prj_mask) in all_projections]
    # compute spectral clustering labels for each sparse graph 
    spectral_clustering_labels = [__compute_spectral(aff_mat,n_clusters=n_clusters, seed=seed) for aff_mat in sparse_graphs]
    # compute CO-cluster matrix using np broadcasting
    CO = (np.array(spectral_clustering_labels)[:, :, None] == np.array(spectral_clustering_labels)[:, None, :]).mean(axis=0)
    # perform sepctral on CO 
    predicted_labels = __consensus_clustering(
        CO,
        num_clusters=n_clusters, 
        seed=seed, 
        threshold=co_threshold
    )

    return predicted_labels


def __generate_projection(data, cat_mask, n_features, rng, p_min_, p_max_):
    """
    the projection is a subspace of the dataframe
    """
    view = data.copy()
    alfa = rng.random()
    selected_features = rng.choice(a=range(n_features), size=round(p_min_+alfa*(p_max_-p_min_)), replace=False)  
    cat_mask_projected = cat_mask[selected_features]
    projected_view = view.iloc[:,selected_features]
    return (projected_view,cat_mask_projected)


def __compute_sparse_similarity(data, cat_mask, k_nn):  
    """
    Compute a sparse graph to encode samples pairwise similarity 

    data: complete data
    cat_mask: categorical mask for Gower distance 
    k_nn: number of k nearest neighbors for sparsity
    TODO mutual: if k_nn has to be applied in a mutual way  
    """  
    gower_dist = gower_matrix(data, cat_features=cat_mask)
    neighbors_idx = np.argsort(gower_dist, axis=1)[:, 1:k_nn+1]

    # build sparse matrix 
    rows = np.repeat(np.arange(gower_dist.shape[0]), k_nn)
    cols = neighbors_idx.flatten()
    A = csr_matrix((gower_dist[rows, cols], (rows, cols)), shape=gower_dist.shape)

    # simmetrize matrix  
    A = 0.5 * (A + A.T)
    # from distance to similairity
    A.data = 1-A.data

    # normalize where each row sums to 1
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # avoid division by zero
    A = A.multiply(1 / row_sums[:, None])
    return A




def __compute_spectral(sparse_affinity_mat,n_clusters,seed):
    """
    Wrapper function to compute spectral clustering from a sparse affinity matrix

    return: labels of clustering
    """
    sc = SpectralClustering(n_clusters=n_clusters,
                            random_state=seed,
                            affinity="precomputed"
                            #assign_labels="cluster_qr"
                            )

    return sc.fit(sparse_affinity_mat).labels_




def __consensus_clustering(CO, num_clusters, seed=None, threshold=0.5):
    """
    This function computes the consensus clustering using the CO-mat as input

    CO: co-matrix (i,j) indicates frequency of i,j being in the same cluster
    threshold: co-association cutoff (e.g., 0.5)
    """
    # threshold
    adj = (CO >= threshold).astype(int)
    graph = csr_matrix(adj)

    sc = SpectralClustering(n_clusters=num_clusters,
                            random_state=seed,
                            affinity="precomputed"
                            #assign_labels="cluster_qr"
                            )
    return sc.fit(graph).labels_
