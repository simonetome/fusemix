"""
Define a class to run a simulation pipeline
"""

from fusemix.amputation import Amputer

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
        multiple amputed data

        """
        self.amputer = Amputer(dataset=self.complete_data,
                                seed=self.seed,
                                prop=self.prop,
                                mnar_freq=self.mnar_freq,
                                mf_proportion=self.mf_proportion)
        self.amputer.generate_amputation()
        self.incomplete_data = self.amputer.incomplete_dataset
        # Don't really know why but required (ERROR) by miceforest
        self.incomplete_data = self.incomplete_data.reset_index(drop=True)

        









