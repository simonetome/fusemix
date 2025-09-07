
k-CMM = https://www.sciencedirect.com/science/article/pii/S0020025521004114?via%3Dihub


Questions:
- does multiple imputation improves clustering?


Choice of datasets
- Parkinsons 174 No missing values 
- breast cancer wisconsin diagnostic 17


With missing data
- dermatology 33
- hearth disease 45
- Breast Cancer Wisconsin (Original) 15
- Myocardial infarction complications 579 Large and with missign values 
- Hepatitis 46 Missing 
- Cirrhosis Patient Survival Prediction 878 Missing
- horse colic 47


## Simulation experiments 

-) Original number of classes as a suggestion for clustering number;
-) Avoid analyzing myocardial complications ID=579, too many problems;
-) For two dataset (heapatitis and horse, ids 46 and 47) I cant drop rows to make a complete dataset.
   a comparison  
-) The other datasets (5 datasets) can be "made complete" by dropping few instances;

### Tested algorithms 

-) Baseline model for comparison
    - Spectral clustering on complete data CCA

#### Generating single imputation
- single imputation from miceforest 
- single imputation from KNN using 10 neighbors

#### Generating multiple imputations 
- imputation using miceforest 
How many imputations? From MICA paper, with M=10 we obtain 95% efficiency.

Models for incomplete data
-) Spectral clustering on single imputation
-) KPOD 
-) Our proposal
X) Ensemble on multiple imputations

MICA does not provide useful code (no singel entry point, no API, no documentation)
kCMM (other incomplete not multiple) poorly documented

### Internal validation metrics comparison

Internal validation metrics can be calculated using the complete observations

### External validation metrics - CCA

-) Calculated against the complete case analysis 
    True labels: CCA assignments 
Can be also calculated using the - Classification analysis
    True labels: target classes


