from keras.constraints import Constraint
import keras.backend as K
import numpy as np

class Between(Constraint):
    def __init__(self, init_value, adj_thres):
        self.adj_thres = adj_thres
        # self.min_value = init_value - self.adj_thres
        self.max_value = init_value + self.adj_thres
        self.min_value = np.clip(init_value - self.adj_thres, 0, init_value)
        # self.max_value = np.clip(init_value + self.adj_thres, init_value, 1) 

    def __call__(self, w): 
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}