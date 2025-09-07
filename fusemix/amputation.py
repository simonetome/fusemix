# -*- coding: utf-8 -*-
"""
Created on Fri May 16 09:48:52 2025

@author: Simone Tomè
"""
import pandas as pd
import numpy as np
from pyampute.ampute import MultivariateAmputation
from scipy.stats import norm
from scipy.special import binom


class Amputer:
    def __init__(self,
                 dataset: pd.DataFrame,
                 prop: float = 0.5,
                 mf_proportion: float = 0.2,
                 mnar_freq: float = 0.1,
                 seed: int = None
                 ):
        """

        Args:
            dataset:
            prop: proportion of missing rows
            mf_proportion: how many features can miss
            mnar_freq: how many features are mnar
            seed:
        """
        self.complete_dataset = dataset
        self.prop = prop
        self.mf_proportion = mf_proportion
        self.mnar_freq = mnar_freq
        self.seed = seed
        # Use a local random generator
        self.rng = np.random.default_rng(seed)
        
    def __generate_patterns(self,num_patterns, length_patterns):
        # Because too many patterns results in problems:
        # randomly keep n patterns (num. samples) if the number of patterns 
        # exceedes that quantity
        num_masks = int(num_patterns)
        mask_length = length_patterns
        masks = self.rng.integers(0, 2, size=(num_masks, mask_length))
        masks = masks[masks.sum(axis=1) > 0]
        return masks
    
    def __assign_pattern_frequency(self, values):
        # the frequency of each pattern is controlled by a gaussian distribution
        mu = np.ceil(self.mf_number/2)  # mean at the center
        sigma = self.mf_number / 6     # controls spread (adjust for sharpness)
        frequencies = norm.pdf([i for i in range(1,self.mf_number+1)], loc=mu, scale=sigma)
        frequencies = dict(zip([i for i in range(1,self.mf_number+1)],frequencies))
        
        freq_patterns = []
        for v in values:
            # divide the frequencies for the number of POSSIBLE patterns
            # with such amount of missing features
            freq_patterns.append(frequencies[v]/(binom(self.mf_number,v)))
        
        # normalize to 1
        freq_patterns = np.array(freq_patterns)
        freq_patterns /= freq_patterns.sum()
        freq_patterns = freq_patterns.tolist()
        
        return freq_patterns
    
    def __build_pyampute_params(self):
        # Build the parameters for pyampute
        parameters = []

        for i,p in enumerate(self.masks):
            params_ = {'incomplete_vars': self.incomplete_features[np.array(p, dtype = bool)].tolist(),
                               'freq': self.freq_patterns[i]}
            if np.any(np.isin(self.mnar_features,params_['incomplete_vars'])):
                params_['mechanism'] = "MNAR"

            parameters.append(params_)
        return parameters

    def generate_amputation(self):
        
        # Step 1: randomly select features to ampute
        features = self.complete_dataset.columns.values.tolist()
        self.mf_number = int(np.ceil(len(features)*self.mf_proportion)) # how many missing features
        self.incomplete_features =  self.rng.choice(features, size = int(self.mf_number), replace=False) # randomly select
        
        # Step 2: generate missing data patterns

        self.masks = self.__generate_patterns(num_patterns = int(np.floor(np.sqrt(self.complete_dataset.shape[0]))),
                                              length_patterns = len(self.incomplete_features))
        self.masks = np.unique(self.masks, axis = 0)

        # Step 3: assign frequencies to each md pattern
        # number of missing features in the patterns
        values = np.sum(self.masks,1)
        self.freq_patterns = self.__assign_pattern_frequency(values)
        
        # Step 4: select the features with MNAR mechanisms
        # All patterns containing MNAR features will be MNAR
        self.mnar_number = int(np.ceil(self.mf_number*self.mnar_freq))
        self.mnar_features = self.rng.choice(self.incomplete_features, 
                                             size = int(self.mnar_number), 
                                             replace=False)


        
        # Step 5: build pyampute parameters
        self.parameters = self.__build_pyampute_params()


        # Step 6: run amputation using pyampute
        ma = MultivariateAmputation(patterns = self.parameters,
                                    prop = self.prop,
                                    seed = self.seed)

        self.incomplete_dataset = ma.fit_transform(self.complete_dataset)
        

    






















