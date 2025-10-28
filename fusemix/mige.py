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