import numpy as np
import pandas as pd

class LDA():
    def __init__(self, dataset, targets):
        if type(dataset) == pd.dataframe:
            #convert to numpy
        self.data = dataset
        self.targets = targets
        
    def calculate_between_class_variance(self):
        
    
    def calculate_within_class_variance(self):
        
    
    def calculate_fishers_criterion(self):
        
        