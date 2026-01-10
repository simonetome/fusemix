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

"""

from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
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
    leiden = False,
    k_nn = 10,
    co_threshold = 0.5,
    mutual = True
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
    if num_projections > 0 and p_min < 1:
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

        sparse_graphs = [__compute_sparse_similarity(prj_view,prj_mask,k_nn,mutual) for (prj_view, prj_mask) in all_projections]
    else:
        # if i don't want to project, I simply use the multiple imputed data
        sparse_graphs = [__compute_sparse_similarity(view,cat_mask,k_nn,mutual) for view in multiple_imputed_data]

    if leiden:
        # use leiden algorithm
        partition_labels = [__community_detection(aff_mat, seed=seed) for aff_mat in sparse_graphs]
    else:    
        # compute spectral clustering labels for each sparse graph 
        partition_labels = [__compute_spectral(aff_mat,n_clusters=n_clusters, seed=seed) for aff_mat in sparse_graphs]
    
    # compute CO-cluster matrix using np broadcasting
    CO = (np.array(partition_labels)[:, :, None] == np.array(partition_labels)[:, None, :]).mean(axis=0)
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


def __compute_sparse_similarity(data, cat_mask, k_nn, mutual=True):
    """
    Compute a sparse symmetric similarity graph using Gower distance.
    """
    # Compute full pairwise Gower distance matrix (dense)
    gower_dist = gower_matrix(data, cat_features=cat_mask)

    # Sort distances row-wise and get k nearest neighbors
    neighbors_idx = np.argsort(gower_dist, axis=1)[:, 1:k_nn+1]  # skip self distance at col 0
    rows = np.arange(gower_dist.shape[0])[:, None]

    A = np.ones_like(gower_dist)
    A[rows, neighbors_idx] = gower_dist[rows, neighbors_idx]
    A[rows,rows] = 0

    if mutual:
        A_sim_ = 1-A
        A_sim = A_sim_ * ((A_sim_ > 0) & (A_sim_.T > 0))
    else:
        A_sim = 1-(1/2*(A + A.T))

    return A_sim





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



def __community_detection(A, sparse = False, seed = None):

    if sparse:
        A_sparse = sp.csr_matrix(A)
        g = ig.Graph.Adjacency(A_sparse.toarray(), mode="UNDIRECTED")
    else:
        g = ig.Graph.Weighted_Adjacency(A, mode="UNDIRECTED", attr="weight")

    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
    return part.membership