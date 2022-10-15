import numpy as np
from constants import CM_FPCMCI
from models.attention.SelfAttention import SelfAttention
from models.attention.InputAttention import InputAttention
from keras.layers import *
from keras.models import *
import tensorflow as tf
import models.Words as W


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
        self.outdense1 = Dense(self.config[W.D1UNITS], activation = self.config[W.D1ACT], name = self.target_var + '_D1')
        self.outdense2 = Dense(self.config[W.D2UNITS], activation = self.config[W.D2ACT], name = self.target_var + '_D2')
        self.out = Dense(self.config[W.NFUTURE], activation = 'linear', name = self.target_var + '_DOUT')
        

    def call(self, x):
        if self.config[W.USEATT]:
            # Attention
            x_selfatt = self.selfatt(x)
            x_inatt = self.inatt([x, self.past_h, self.past_c])

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
            dec = self.dec(repeat, initial_state = [h, c])
        else:
            dec = self.dec(repeat)

        y = Dropout(0.5)(dec)
        y = self.outdense1(y)
        y = Dropout(0.5)(y)
        y = self.outdense2(y)
        y = Dropout(0.5)(y)
        y = self.out(y)
        y = tf.expand_dims(y, axis = -1)

        return y