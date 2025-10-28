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



























