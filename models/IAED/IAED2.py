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
        self.searchBest = searchBest

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
            self.selfenc1 = LSTM(int(self.config[W.ENCDECUNITS]/2), 
                                 name = target_var + '_selfENC1',
                                 return_sequences = True,
                                 input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
            self.selfenc2 = LSTM(int(self.config[W.ENCDECUNITS]/2), 
                                 name = target_var + '_selfENC2',
                                 return_state = True,
                                 input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))

            self.inenc1 = LSTM(int(self.config[W.ENCDECUNITS]/2),
                               name = target_var + '_inENC1',
                               return_sequences = True,
                               input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
            self.inenc2 = LSTM(int(self.config[W.ENCDECUNITS]/2),
                               name = target_var + '_inENC2',
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
            self.enc1 = LSTM(self.config[W.ENCDECUNITS], 
                             name = target_var + '_ENC1',
                             return_sequences = True,
                             input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
            self.enc2 = LSTM(self.config[W.ENCDECUNITS], 
                             name = target_var + '_ENC2',
                             return_state = True,
                             input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))

        self.repeat = RepeatVector(self.config[W.NFUTURE], name = self.target_var + '_REPEAT')

        # Decoder
        self.dec1 = LSTM(self.config[W.ENCDECUNITS], return_sequences = True, name = self.target_var + '_DEC1')
        self.dec2 = LSTM(self.config[W.ENCDECUNITS], name = self.target_var + '_DEC2')

        # Dense
        # self.outdense1 = DenseDropout(self.config[W.NFUTURE] * 3, self.config[W.D1ACT], self.config[W.DRATE])
        self.outdense = DenseDropout(self.config[W.NFUTURE] * 2, self.config[W.DACT], self.config[W.DRATE])
        self.out = DenseDropout(self.config[W.NFUTURE], 'linear', 0)
        

    def call(self, x):
        if self.config[W.USEATT]:
            # Attention
            x_selfatt = self.selfatt(x)
            # if not self.searchBest: x_selfatt = Dropout(self.config[W.DRATE])(x_selfatt)

            x_inatt = self.inatt([x, self.past_h, self.past_c])
            # if not self.searchBest: x_inatt = Dropout(self.config[W.DRATE])(x_inatt)

            # Encoders
            enc1_1 = self.selfenc1(x_selfatt)
            enc2_1 = self.inenc1(x_inatt)
            enc1_2, h1, c1 = self.selfenc2(enc1_1)
            enc2_2, h2, c2 = self.inenc2(enc2_1)
            self.past_h.assign(tf.expand_dims(h2[-1], -1))
            self.past_c.assign(tf.expand_dims(c2[-1], -1))

            x = concatenate([enc1_2, enc2_2])
            if self.config[W.DECINIT]:
                h = concatenate([h1, h2])
                c = concatenate([c1, c2])
        else:
            x = self.enc1(x)
            x, h, c = self.enc2(x)
            
        repeat = self.repeat(x)
            
        # Decoder
        if self.config[W.DECINIT]:
            y = self.dec1(repeat, initial_state = [h, c])
        else:
            y = self.dec1(repeat)
        y = self.dec2(y)

        if not self.searchBest: y = Dropout(self.config[W.DRATE])(y)
        # y = self.outdense1(y)
        y = self.outdense(y)
        y = self.out(y)
        y = tf.expand_dims(y, axis = -1)

        return y