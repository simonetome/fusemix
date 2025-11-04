"""
Implementation of MICA algorithm

Paper: Incomplete clustering analysis via multiple imputation
https://www.tandfonline.com/doi/full/10.1080/02664763.2022.2060952

Steps:

1. Multiple imputation on dataset nxp
2. Clustering on each of the M imputed view, k_j can be different between views
3. Being K the total amount of clusters build the Kxp matrix of all centroids. 
   Perform clustering on centroids matrix and re-calculate centroids. 
4. Assign an observation to the cluster centroid based on majority voting.

"""
import numpy as np

from sklearn.cluster import KMeans

def compute_MICA(
   multiple_imputed_data,
   num_clusters,
   seed
   ):
   """Function to compute MICA clustering

   Args:
       multiple_imputed_data (list[pandas.DataFrame]): multiple imputed dataset  
       num_clusters (integer): how many clusters
       seed (integer): integer seed for reproducibility

   Returns:
       ArrayLike: list of cluster labels
   """
   
   results = []
   for view in multiple_imputed_data: # view is a imputed dataset 
      kmeans = KMeans(init="k-means++", 
                      n_clusters=num_clusters,
                      n_init=1, # I set init 1 to enforce some diveristy across the ensembles
                      random_state=seed)
      results += [kmeans.fit(view).cluster_centers_]

   # clustering centroids 
   kmean_centroids = KMeans(init="k-means++", 
                            n_clusters=num_clusters,
                            n_init="auto", 
                            random_state=seed)
   kmean_centroids = kmean_centroids.fit(np.vstack(results))

   # For each view: assign nearest centroids (recalculated centroids from 2nd kmeans)
   predictions = [kmean_centroids.predict(X=view) for view in multiple_imputed_data]

   # majority voting
   majority_votes = np.array([
      np.bincount(col).argmax() for col in np.vstack(predictions).T
   ])

   return majority_votes

























