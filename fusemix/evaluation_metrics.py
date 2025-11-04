"""
 A collection of helper functions to compute internal and external 
 validation metrics in clustering. Functions are wrappers to scikit-learn.

Functions:
 external_metrics: compare predicted labels with a ground truth
 internal_metrics: evaluate predicted labels in terms of clustering properties

"""

from gower import gower_matrix
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, completeness_score, v_measure_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


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


def internal_metrics(predicted_labels, complete_data, cat_mask):
    """
    internal validation metrics are tailored to clustering and can evaluate 
    different qualities of a clustering results, e.g., separability, compactness, etc.

    there are no true labels

    sil -> higher means better
    ch -> higher means better
    db -> lower means better

    """
    gower_dist = gower_matrix(complete_data,cat_features = cat_mask)
    sil_score = silhouette_score(X=gower_dist,metric="precomputed",labels=predicted_labels)
    dbouldin_score = davies_bouldin_score(X=complete_data, labels=predicted_labels)
    charabasz_score = calinski_harabasz_score(X=complete_data, labels=predicted_labels)

    return {'sh':sil_score,
            'ch':charabasz_score,
            'db':dbouldin_score}

