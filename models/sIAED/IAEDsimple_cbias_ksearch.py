import numpy as np
from constants import CM_FPCMCI
from models.attention.SelfAttention_cbias_ksearch import SelfAttention
from models.attention.InputAttention import InputAttention
from keras.layers import *
from keras.models import *
import tensorflow as tf
from models.words import *
import keras.backend as K


class IAED(Layer):
    def __init__(self, config, target_var, name = "IAED"):
        super(IAED, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        if self.config[W_USEATT]:

            # Causal vector definition
            causal_vec = np.array(CM_FPCMCI[self.config[W_FEATURES].index(self.target_var), :]) if self.config[W_USECAUSAL] else None
            
            # Self attention
            self.selfatt = SelfAttention(self.config, causal_vec, name = self.target_var + '_selfatt')
            
            # Input attention
            self.inatt = InputAttention(self.config, name = self.target_var + '_inatt')
           
            # Encoders
            self.selfenc = LSTM(int(self.config["ENCDECUNITS"]/2), 
                                name = target_var + '_selfENC',
                                return_state = True,
                                input_shape = (self.config[W_NPAST], self.config[W_NFEATURES]))

            self.inenc = LSTM(int(self.config["ENCDECUNITS"]/2),
                              name = target_var + '_inENC',
                              return_state = True,
                              input_shape = (self.config[W_NPAST], self.config[W_NFEATURES]))

            # Initialization
            self.past_h = tf.Variable(tf.zeros([int(self.config["ENCDECUNITS"]/2), 1]), 
                                                trainable = False, 
                                                shape = (int(self.config["ENCDECUNITS"]/2), 1),
                                                name = self.target_var + '_pastH')
            self.past_c = tf.Variable(tf.zeros([int(self.config["ENCDECUNITS"]/2), 1]), 
                                                trainable = False, 
                                                shape = (int(self.config["ENCDECUNITS"]/2), 1),
                                                name = self.target_var + '_pastC')

        else:
            self.enc = LSTM(self.config["ENCDECUNITS"], 
                            name = target_var + '_ENC',
                            return_state = True,
                            input_shape = (self.config[W_NPAST], self.config[W_NFEATURES]))

        self.repeat = RepeatVector(self.config[W_NFUTURE], name = self.target_var + '_REPEAT')

        # Decoder
        self.dec = LSTM(self.config["ENCDECUNITS"], name = self.target_var + '_DEC')

        # Dense
        self.outdense1 = Dense(self.config["D1UNITS"], activation = self.config["D1ACT"], name = self.target_var + '_D')
        self.outdense2 = Dense(self.config["D2UNITS"], activation = self.config["D2ACT"], name = self.target_var + '_D')
        self.out = Dense(self.config[W_NFUTURE], activation = 'linear', name = self.target_var + '_out')
        

    def call(self, x):
        if self.config[W_USEATT]:
            # Attention
            x_selfatt = self.selfatt(x)
            x_inatt = self.inatt([x, self.past_h, self.past_c])

            # Encoders
            enc1, h1, c1 = self.selfenc(x_selfatt)
            enc2, h2, c2 = self.inenc(x_inatt)
            self.past_h.assign(tf.expand_dims(h2[-1], -1))
            self.past_c.assign(tf.expand_dims(c2[-1], -1))

            x = concatenate([enc1, enc2])
            if self.config["DECINIT"]:
                h = concatenate([h1, h2])
                c = concatenate([c1, c2])
        else:
            x, h, c = self.enc(x)
            
        repeat = self.repeat(x)
            
        # Decoder
        if self.config["DECINIT"]:
            dec = self.dec(repeat, initial_state = [h, c])
        else:
            dec = self.dec(repeat)

        y = self.outdense1(dec)
        y = self.outdense2(y)
        y = self.out(y)
        y = tf.expand_dims(y, axis = -1)

        return y