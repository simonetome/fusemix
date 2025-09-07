# -*- coding: utf-8 -*-

import miceforest as mf
import pandas as pd
from datetime import datetime

class MultipleImputer:
    
    def __init__(self, 
                 incomplete_data: pd.DataFrame,
                 mean_match_strategy: str,
                 seed: int = None,
                 num_imputations: int = 5,
                 mean_match_candidates: int = 10
                 ):
        
        self.kernel = mf.ImputationKernel(
            incomplete_data,
            random_state=seed,
            num_datasets=num_imputations,
            mean_match_candidates=mean_match_candidates,
            mean_match_strategy=mean_match_strategy
        )
        self.num_imputations = num_imputations
        self.complete_data = None
        
    def run_mice(self, **kwargs):
        start_t = datetime.now()
        self.complete_data = self.kernel.mice(**kwargs)
        end_t = datetime.now()
        self.time_ = (end_t - start_t).total_seconds()
        
    def get_dataset(self, num: int = 1):
        return self.kernel.complete_data(dataset = num)

    def get_multiple_imputations(self):
        return [self.kernel.complete_data(dataset = i) for i in range(self.num_imputations)]
        























