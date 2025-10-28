from pathlib import Path
import os,sys
import pickle
import warnings
import importlib
import pandas as pd
import time

from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fusemix.imputation import MultipleImputer

warnings.filterwarnings("ignore")

BASE_PATH = os.path.abspath(os.path.dirname(sys.argv[0]))    
TEST_DATA_PATH = os.path.normpath(os.path.join(BASE_PATH, "../test_data"))
TEST_OUT_PATH = os.path.normpath(os.path.join(BASE_PATH, "../test_output"))

print(f"base path {BASE_PATH}")
print(f"test path {TEST_DATA_PATH}")
print(f"test out path {TEST_OUT_PATH}")

def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def write_pickle(var, path):
    with open(path, 'wb') as f:
       pickle.dump(var, f)

ids = pd.read_csv(os.path.join(TEST_OUT_PATH,"datasets_analysis.csv"))['Id']

n_runs = 10

md_param_grid = {
    'props': [0.75,1.],
    'mf_proportions': [0.5,0.75],
    'mnar_proportions': [0.,0.25,0.5]
}


# create directories
for id in ids:
    # create directory for dataset id in test_data/missing_data
    directory = os.path.join(TEST_DATA_PATH,"imputed_data/"+str(id))
    if not os.path.exists(directory):
        os.makedirs(directory)
    # create directory for dataset id in test_data/missing_data/parameters
    for prop in md_param_grid['props']:
        for mf_proportion in md_param_grid['mf_proportions']:
            for mnar_proportion in md_param_grid['mnar_proportions']:
                directory = os.path.join(TEST_DATA_PATH,"imputed_data/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion))
                if not os.path.exists(directory):
                    os.makedirs(directory)


# MICE imputation
for id in tqdm(ids,desc="Dataset processed"):
    for prop in md_param_grid['props']:
        for mf_proportion in md_param_grid['mf_proportions']:
            for mnar_proportion in md_param_grid['mnar_proportions']:
                
                # Load amputed data 
                directory = os.path.join(TEST_DATA_PATH,"missing_data/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion))

                for seed in tqdm(range(n_runs), desc="Random seeds", leave=False):
                    missing_data_test = read_pickle(os.path.join(directory,"data_pipeline_"+str(seed)+".pkl"))

                    mice_imputer = MultipleImputer(incomplete_data=missing_data_test.incomplete_data, 
                                seed = seed,
                                num_imputations=10,
                                mean_match_candidates=0,
                                mean_match_strategy = "Normal")
                    #mice_imputer.run_mice(iterations=2, num_estimators = 50)
                    mice_imputer.run_mice(iterations=2)
                    imputed_data = mice_imputer.get_multiple_imputations()
                    directory_output = os.path.join(TEST_DATA_PATH,"imputed_data/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion))
                    write_pickle(imputed_data, os.path.join(directory_output,"data_imputed_"+str(seed)+".pkl"))

                time.sleep(10)


































