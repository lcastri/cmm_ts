import numpy as np
from constants import CM_FPCMCI
from models.attention.SelfAttention import SelfAttention
from keras.layers import *
from keras.models import *
import models.Words as W
from models.DenseDropout import DenseDropout

class CNNLSTM(Layer):
    def __init__(self, config, target_var, name = "CNNLSTM", searchBest = False):
        super(CNNLSTM, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        if self.config[W.USEATT]:
            # Causal vector definition
            if searchBest:
                causal_vec = CM_FPCMCI[0, :] if self.config[W.USECAUSAL] else None
            else:
                causal_vec = np.array(self.config[W.CMATRIX][self.config[W.FEATURES].index(self.target_var), :]) if self.config[W.USECAUSAL] else None
            
            # Self attention
            self.selfatt = SelfAttention(self.config, causal_vec, name = self.target_var + '_selfatt')
           
        # Convs
        self.conv1 = Conv1D(filters = 64, kernel_size = 2, activation = 'relu')
        self.conv2 = Conv1D(filters = 32, kernel_size = 2, activation = 'relu')
        self.maxpool = MaxPooling1D(pool_size = 2)
        self.flatten = Flatten()
        self.repeat = RepeatVector(self.config[W.NFUTURE])
        self.rnn = LSTM(100, activation='tanh')
        # Dense
        self.outdense = DenseDropout(self.config[W.NFUTURE] * 2, self.config[W.DACT], self.config[W.DRATE])
        self.out = DenseDropout(self.config[W.NFUTURE], 'linear', 0)



    def call(self, x):
        # Input attention
        if self.config[W.USEATT]: x = self.selfatt(x)

        # Convs
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.maxpool(y)
        y = self.flatten(y)
        y = self.repeat(y)

        # LSTM
        y = self.rnn(y)

        # Dense
        y = self.outdense(y)
        y = self.out(y)

        return y