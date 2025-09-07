"""
Define a class to run a simulation pipeline

"""

from sklearn.impute import KNNImputer
from fusemix.imputation import MultipleImputer
from fusemix.amputation import Amputer
from sklearn.cluster import spectral_clustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from gower import gower_matrix

import snf
import numpy as np
import pandas as pd

from dataclasses import dataclass


@dataclass
class DataPipelineConfig:
    """
    used to configure the pipeline to generate data
    """
    seed: int = 10
    prop: float = 0.1
    mnar_freq : float = 0.1
    mf_proportion: float = 0.1
    mice_iterations: int = 2
    num_imputations: int = 10
    cat_mask: list = None
    complete_data: pd.DataFrame = None
    num_classes: int = None
    verbose: bool = True

class PipelineDataGeneration:
    """    
    generate data for experiments
    """
    def __init__(self, cfg: DataPipelineConfig):

        # parameters of the configuration
        self.seed = cfg.seed
        self.prop = cfg.prop
        self.mnar_freq = cfg.mnar_freq
        self.mf_proportion = cfg.mf_proportion
        self.mice_iterations = cfg.mice_iterations
        self.num_imputations = cfg.num_imputations
        self.cat_mask = cfg.cat_mask
        self.complete_data = cfg.complete_data
        self.verbose = cfg.verbose
        self.num_classes = cfg.num_classes

        self.amputer = None
        self.imputer = None
        self.incomplete_data = None
        self.multiple_data = None
        
        
    def run(self):
        """
        Run the pipeline
        
        Returns:
        multiple imputed data

        """
        if self.verbose:
            print("=================================================")
            print("Running pipeline")
            print("=================================================")
            print("Step1: Amputation")
        self.amputer = Amputer(dataset=self.complete_data,
                                seed=self.seed,
                                prop=self.prop,
                                mnar_freq=self.mnar_freq,
                                mf_proportion=self.mf_proportion)
        self.amputer.generate_amputation()
        self.incomplete_data = self.amputer.incomplete_dataset
        # Don't really know why but required (ERROR) by miceforest
        self.incomplete_data = self.incomplete_data.reset_index(drop=True)

        




class Pipeline:

    def __init__(self, pipeline_params, verbose=False):

        self.complete_data = pipeline_params['complete_data']

        self.true_labels = pipeline_params['y_data']

        # encode labels
        self.unique_categories,  self.true_labels = np.unique(self.true_labels, return_inverse=True)

        self.true_labels =  self.true_labels.flatten()

        self.seed = pipeline_params['seed']
        self.prop = pipeline_params['prop']
        self.mnar_freq = pipeline_params['mnar_freq']
        self.mf_proportion = pipeline_params['mf_proportion']
        self.mice_iterations = pipeline_params['mice_iterations']
        self.num_imputations = pipeline_params['num_imputations']
        self.cat_mask = pipeline_params['cat_mask']
        self.nn_snf = pipeline_params['nn_snf']
        self.k = len(self.unique_categories)

        self.verbose = verbose

        self.amputer = None
        self.imputer = None
        self.incomplete_data = None
        self.multiple_data = None
        self.affinities = None
        self.fused_networks = None

        self.fusemix_labels = None
        self.complete_labels = None
        self.knn_labels = None

        self.complete_vs_true = {}
        self.fuse_vs_true = {}
        self.fuse_vs_complete = {}
        self.knn_vs_true = {}
        self.knn_vs_complete = {}


    def run(self):
        """
        Run the pipeline
        Returns:

        """
        if self.verbose:
            print("=================================================")
            print("Running pipeline")
            print("=================================================")
            print("Step1: Amputation")
        self.amputer = Amputer(dataset=self.complete_data,
                                seed=self.seed,
                                prop=self.prop,
                                mnar_freq=self.mnar_freq,
                                mf_proportion=self.mf_proportion)
        self.amputer.generate_amputation()
        self.incomplete_data = self.amputer.incomplete_dataset

        if self.verbose:
            print("=================================================")
            print("Step2: Imputation")
        
        self.imputer = MultipleImputer(incomplete_data = self.incomplete_data,
                                       num_imputations = self.num_imputations,
                                       seed = self.seed)
        self.imputer.run_mice(self.mice_iterations)
        self.multiple_data = self.imputer.get_multiple_imputations()
        if self.verbose:
            print("=================================================")
            print("Step3: FuseMIX")
        self.affinities = [snf.compute.make_affinity(d,
                                                     metric = "gower",
                                                     K = self.nn_snf,
                                                     cat_features = self.cat_mask) for d in self.multiple_data]
        self.fused_network = snf.snf(self.affinities, K=self.nn_snf)

        # Fusemix Labels
        self.fusemix_labels = spectral_clustering(affinity=self.fused_network,
                                                  n_clusters=self.k, random_state=self.seed)
        if self.verbose:
            print("=================================================")
            print("Step3: Complete clustering")
        # Complete labels
        gower_dist_complete = gower_matrix(self.complete_data,
                                           cat_features=self.cat_mask)
        self.complete_labels = spectral_clustering(affinity=(1 - gower_dist_complete),
                                                    n_clusters=self.k,
                                                    random_state=self.seed)
        if self.verbose:
            print("=================================================")
            print("Step3: KNN clustering")

        # Naive imputation and clustering
        X_imputed = KNNImputer().fit_transform(self.incomplete_data)
        gower_dist_knn = gower_matrix(X_imputed, cat_features=self.cat_mask)
        self.knn_labels = spectral_clustering(affinity=1 - gower_dist_knn,
                                              n_clusters=self.k,
                                              random_state=self.seed)

        # Clustering Fusemix vs true classes
        self.fuse_vs_true['ARI'] = adjusted_rand_score(self.true_labels,
                                                       self.fusemix_labels)
        self.fuse_vs_true['AMI'] = adjusted_mutual_info_score(self.true_labels,
                                                              self.fusemix_labels)

        # Fusemix vs complete clustering
        self.fuse_vs_complete['ARI'] = adjusted_rand_score(self.complete_labels,
                                                           self.fusemix_labels)
        self.fuse_vs_complete['AMI'] = adjusted_mutual_info_score(self.complete_labels,
                                                                  self.fusemix_labels)

        # Complete vs true classes
        self.complete_vs_true['ARI'] = adjusted_rand_score(self.true_labels,
                                                           self.complete_labels)
        self.complete_vs_true['AMI'] = adjusted_mutual_info_score(self.true_labels,
                                                                  self.complete_labels)

        # Knn vs true
        self.knn_vs_true['ARI'] = adjusted_rand_score(self.true_labels,
                                                      self.knn_labels)
        self.knn_vs_true['AMI'] = adjusted_mutual_info_score(self.true_labels,
                                                             self.knn_labels)

        # Knn vs complete
        self.knn_vs_complete['ARI'] = adjusted_rand_score(self.complete_labels,
                                                          self.knn_labels)
        self.knn_vs_complete['AMI'] = adjusted_mutual_info_score(self.complete_labels,
                                                                 self.knn_labels)

        if self.verbose:
            print("=================================================")
            print("Completed")
            print("=================================================")













