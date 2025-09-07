# -*- coding: utf-8 -*-

from fusemix.amputation import Amputer 
from fusemix.imputation import MultipleImputer

from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score

import snf


SEED = 2

# import dataset
dataset = fetch_ucirepo(id=17)
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = dataset.data.features
y = dataset.data.targets

# select categorical variables
cat_variables = dataset.variables.loc[dataset.variables['type'] == 'Categorical', 'name']
cat_variables_in_X = cat_variables[cat_variables.isin(X.columns)]
cat_mask = X.columns.isin(cat_variables_in_X)


amputer = Amputer(dataset = X, seed = SEED)
amputer.generate_amputation()
md1 = amputer.incomplete_dataset

amputer_2 = Amputer(dataset = X, seed = SEED)
amputer_2.generate_amputation()
md2 = amputer_2.incomplete_dataset

imputer = MultipleImputer(incomplete_data = md1,num_imputations = 10,seed = SEED)
imputer.run_mice(1)
imputer.time_




multiple_data = imputer.get_multiple_imputations()
affinities = [snf.compute.make_affinity(d,metric = "gower", K = 10) for d in imputer.get_multiple_imputations()]
fused_network = snf.snf(affinities, K = 10)

labels = spectral_clustering(fused_network, n_clusters=2)

unique_categories, ordinal_encoded = np.unique(y, return_inverse=True)

print("Original categories:", y)
print("Ordinal encoding:", ordinal_encoded)
print("Category mapping:", dict(enumerate(unique_categories)))

# check between labels and predictions
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(ordinal_encoded, labels)
plt.show()















