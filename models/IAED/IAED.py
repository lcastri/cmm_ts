from matplotlib.pyplot import yscale
import numpy as np
from constants import CM_FPCMCI
from models.attention.SelfAttention import SelfAttention
from models.attention.InputAttention import InputAttention
from keras.layers import *
from keras.models import *
import tensorflow as tf
import models.Words as W
from models.DenseDropout import DenseDropout


class IAED(Layer):
    def __init__(self, config, target_var, name = "IAED", searchBest = False):
        super(IAED, self).__init__(name = name)
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
            
            # Input attention
            self.inatt = InputAttention(self.config, name = self.target_var + '_inatt')
           
            # Encoders
            self.selfenc = LSTM(int(self.config[W.ENCDECUNITS]/2), 
                                name = target_var + '_selfENC',
                                return_state = True,
                                input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))

            self.inenc = LSTM(int(self.config[W.ENCDECUNITS]/2),
                              name = target_var + '_inENC',
                              return_state = True,
                              input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))

            # Initialization
            self.past_h = tf.Variable(tf.zeros([int(self.config[W.ENCDECUNITS]/2), 1]), 
                                                trainable = False, 
                                                shape = (int(self.config[W.ENCDECUNITS]/2), 1),
                                                name = self.target_var + '_pastH')
            self.past_c = tf.Variable(tf.zeros([int(self.config[W.ENCDECUNITS]/2), 1]), 
                                                trainable = False, 
                                                shape = (int(self.config[W.ENCDECUNITS]/2), 1),
                                                name = self.target_var + '_pastC')

        else:
            self.enc = LSTM(self.config[W.ENCDECUNITS], 
                            name = target_var + '_ENC',
                            return_state = True,
                            input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))

        self.repeat = RepeatVector(self.config[W.NFUTURE], name = self.target_var + '_REPEAT')

        # Decoder
        self.dec = LSTM(self.config[W.ENCDECUNITS], name = self.target_var + '_DEC')

        # Dense
        # self.outdense1 = DenseDropout(self.config[W.NFUTURE] * 3, self.config[W.D1ACT], self.config[W.DRATE])
        self.outdense = DenseDropout(self.config[W.NFUTURE] * 2, self.config[W.DACT], self.config[W.DRATE])
        self.out = DenseDropout(self.config[W.NFUTURE], 'linear', 0)
        

    def call(self, x):
        if self.config[W.USEATT]:
            # Attention
            x_selfatt = self.selfatt(x)
            # x_selfatt = Dropout(self.config[W.DRATE])(x_selfatt)

            x_inatt = self.inatt([x, self.past_h, self.past_c])
            # x_inatt = Dropout(self.config[W.DRATE])(x_inatt)

            # Encoders
            enc1, h1, c1 = self.selfenc(x_selfatt)
            enc2, h2, c2 = self.inenc(x_inatt)
            self.past_h.assign(tf.expand_dims(h2[-1], -1))
            self.past_c.assign(tf.expand_dims(c2[-1], -1))

            x = concatenate([enc1, enc2])
            if self.config[W.DECINIT]:
                h = concatenate([h1, h2])
                c = concatenate([c1, c2])
        else:
            x, h, c = self.enc(x)
            
        repeat = self.repeat(x)
            
        # Decoder
        if self.config[W.DECINIT]:
            y = self.dec(repeat, initial_state = [h, c])
        else:
            y = self.dec(repeat)

        y = Dropout(self.config[W.DRATE])(y)
        # y = self.outdense1(y)
        y = self.outdense(y)
        y = self.out(y)
        y = tf.expand_dims(y, axis = -1)

        return y