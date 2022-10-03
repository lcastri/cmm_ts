import numpy as np
from models.attention.SelfAttention import SelfAttention
from models.attention.InputAttention import InputAttention
from keras.layers import *
from keras.models import *
import tensorflow as tf
from models.words import *


class IAED(Layer):
    def __init__(self, config, target_var, name = "IAED"):
        super(IAED, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        if self.config[W_SETTINGS][W_USEATT]:

            # Attention
            causal_vec = np.array(self.config[W_INPUTATT][W_CMATRIX][self.config[W_SETTINGS][W_FEATURES].index(self.target_var), :]) if self.config[W_INPUTATT][W_USECAUSAL] else None
            self.selfatt = SelfAttention(self.config, causal_vec, name = self.target_var + '_selfatt')
            self.inatt = InputAttention(self.config, causal_vec, name = self.target_var + '_inatt')
           
            # Encoders
            self.selfenc = LSTM(self.config[W_ENC][0][W_UNITS], 
                                name = target_var + '_selfenc',
                                return_sequences = self.config[W_ENC][0][W_RSEQ],
                                return_state = self.config[W_ENC][0][W_RSTATE],
                                input_shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
            self.inenc = LSTM(self.config[W_ENC][0][W_UNITS], 
                            name = target_var + '_inenc',
                            return_sequences = self.config[W_ENC][0][W_RSEQ],
                            return_state = True,
                            input_shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))

            # Initialization
            self.past_h = tf.Variable(tf.zeros([self.config[W_ENC][-1][W_UNITS], 1]), 
                                                trainable = False, 
                                                shape = (self.config[W_ENC][-1][W_UNITS], 1),
                                                name = self.target_var + '_pastH')
            self.past_c = tf.Variable(tf.zeros([self.config[W_ENC][-1][W_UNITS], 1]), 
                                                trainable = False, 
                                                shape = (self.config[W_ENC][-1][W_UNITS], 1),
                                                name = self.target_var + '_pastC')

        else:
            self.enc = LSTM(self.config[W_ENC][0][W_UNITS], 
                            name = target_var + '_enc',
                            return_sequences = self.config[W_ENC][0][W_RSEQ],
                            return_state = self.config[W_ENC][0][W_RSTATE],
                            input_shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))

        self.repeat = RepeatVector(self.config[W_SETTINGS][W_NFUTURE], name = self.target_var + '_REPEAT')

        # Decoder
        self.decs = list()
        for i in range(len(self.config[W_DEC])):
            self.decs.append(LSTM(self.config[W_DEC][i][W_UNITS], 
                                  name = self.target_var + '_DEC',
                                  return_sequences = self.config[W_DEC][i][W_RSEQ]))

        # Dense
        self.outdense = list()
        for i in range(len(self.config[W_OUT])):
            self.outdense.append(Dense(self.config[W_OUT][i][W_UNITS], activation = self.config[W_OUT][i][W_ACT], 
                                       name = self.target_var + '_D'))
        self.out = Dense(self.config[W_SETTINGS][W_NFUTURE], activation = 'linear', name = self.target_var + '_out')
        

    def call(self, x):
        if self.config[W_SETTINGS][W_USEATT]:
            # Attention
            x_selfatt = self.selfatt(x)
            x_inatt = self.inatt([x, self.past_h, self.past_c])

            # Encoders
            h1 = self.selfenc(x_selfatt)
            h2, h, c = self.inenc(x_inatt)
            self.past_h.assign(tf.expand_dims(h[0], -1))
            self.past_c.assign(tf.expand_dims(c[0], -1))

            x = concatenate([h1, h2])
        else:
            x = self.enc(x)
            
        repeat = self.repeat(x)
            
        # Decoder
        dec = repeat
        for i in range(len(self.config[W_DEC])):
            dec = self.decs[i](dec)

        y = dec
        for i in range(len(self.config[W_OUT])):
            y = self.outdense[i](y)
        y = self.out(y)
        y = tf.expand_dims(y, axis = -1)

        return y