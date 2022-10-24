
from keras.layers import *
from models.T2V.T2V import T2V
from constants import CM_FPCMCI
from models.attention.SelfAttention import SelfAttention
from models.attention.InputAttention import InputAttention
import numpy as np
import tensorflow as tf
import models.Words as W


class T2VRNN(Layer):
    
    def __init__(self, config, target_var, name = "T2VRNN", searchBest = False):
        super(T2VRNN, self).__init__(name = name)
        self.config = config
        self.target_var = target_var
        self.searchBest = searchBest

        if self.config[W.USEATT]:
            # Causal vector definition
            if self.searchBest:
                causal_vec = CM_FPCMCI[0, :] if self.config[W.USECAUSAL] else None
            else:
                causal_vec = np.array(self.config[W.CMATRIX][self.config[W.FEATURES].index(self.target_var), :]) if self.config[W.USECAUSAL] else None
            
            # Self attention
            self.selfatt = SelfAttention(self.config, causal_vec, name = self.target_var + '_selfatt')

        # T2V
        self.t2v = T2V(config[W.T2VUNITS], name = self.target_var + '_t2v')

        # RNN block
        self.rnn1 = LSTM(self.config[W.ENCDECUNITS], return_sequences = True, activation = 'tanh', name = self.target_var + '_lstm1')
        self.rnn2 = LSTM(self.config[W.ENCDECUNITS], activation = 'tanh', name = self.target_var + '_lstm2')

        # Dense
        self.outdense1 = Dense(self.config[W.D1UNITS], activation = self.config[W.D1ACT], name = self.target_var + '_D1')
        self.outdense2 = Dense(self.config[W.D2UNITS], activation = self.config[W.D2ACT], name = self.target_var + '_D2')
        self.outdense3 = Dense(self.config[W.D3UNITS], activation = self.config[W.D3ACT], name = self.target_var + '_D3')
        self.out = Dense(self.config[W.NFUTURE], activation = 'linear', name = self.target_var + '_DOUT')
        

        
    def call(self, x):
        if self.config[W.USEATT]:
            # Attention
            x = self.selfatt(x)
            # x = Dropout(self.config[W.DRATE])(x)

        y = self.t2v(x)
        y = self.rnn1(y)      
        y = self.rnn2(y)      

        # if not self.searchBest: y = Dropout(self.config[W.DRATE])(y)
        y = self.outdense1(y)
        # if not self.searchBest: y = Dropout(self.config[W.DRATE])(y)
        y = self.outdense2(y)
        # if not self.searchBest: y = Dropout(self.config[W.DRATE])(y)
        y = self.outdense3(y)
        # if not self.searchBest: y = Dropout(self.config[W.DRATE])(y)
        y = self.out(y)
        y = tf.expand_dims(y, axis = -1)

        return y
