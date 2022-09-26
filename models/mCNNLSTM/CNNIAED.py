import numpy as np
from .CNNAttention import CNNAttention
from keras.layers import *
from keras.models import *
from models.words import *


class CNNIAED(Layer):
    def __init__(self, config, target_var, name = "IAED"):
        super(CNNIAED, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            self.ca = CNNAttention(self.config, 
                                   np.array(self.config[W_INPUTATT][W_CMATRIX][self.config[W_SETTINGS][W_FEATURES].index(self.target_var), :]), 
                                   name = self.target_var + '_CA')
           
        # Convs
        self.convs = list()
        for i in range(len(self.config[W_CNN])):
                self.convs.append(Conv1D(filters = self.config[W_CNN][i][W_FILTERS], 
                                         kernel_size = self.config[W_CNN][i][W_KSIZE],
                                         activation = self.config[W_CNN][i][W_ACT]))

        self.flatten = Flatten()


    def call(self, x):
        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            x_tilde = self.ca(x)

        # Convs
        y = x_tilde if self.config[W_SETTINGS][W_USEATT] else x
        for i in range(len(self.config[W_CNN])):
            y = self.convs[i](y)

        # Flatten
        y = self.flatten(y)

        return y
