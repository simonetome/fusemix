"""
Script to fetch data from UCI repository
"""

import sys
import os
import pandas as pd
from data_loading import load_uciml
import pickle
      

notebook = False
       
BASE_PATH = os.path.abspath(os.path.dirname(sys.argv[0]))

if notebook:
    BASE_PATH = os.getcwd()
    
FETCHED_PATH = os.path.normpath(os.path.join(BASE_PATH, "../test_data/fetched"))

dataset_ids_path = os.path.join(FETCHED_PATH,"dataset_ids.csv")
dataset_ids = pd.read_csv(dataset_ids_path, sep = " ")
print(dataset_ids.head(3))

datasets = dict.fromkeys(dataset_ids['Id'])

for id in dataset_ids['Id']:
    print(f"Collecting dataset {id}")
    datasets[id] = load_uciml(id=id)


for d in datasets:
    print(os.path.join(FETCHED_PATH,"dataset_"+str(d)+".pkl"))
    with open(os.path.join(FETCHED_PATH,"dataset_"+str(d)+".pkl"),"wb") as out:
        pickle.dump(datasets[d],out)