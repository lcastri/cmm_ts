import numpy as np
# from models.SimpleAttention import CAttention
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

        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            causal_vec = np.array(self.config[W_INPUTATT][W_CMATRIX][self.config[W_SETTINGS][W_FEATURES].index(self.target_var), :]) if self.config[W_INPUTATT][W_USECAUSAL] else None
            self.ca = InputAttention(self.config, 
                                 causal_vec, 
                                 name = self.target_var + '_CA')
           
        # Encoders
        self.encs = list()
        for i in range(len(self.config[W_ENC])):
            if i == 0:
                self.encs.append(LSTM(self.config[W_ENC][i][W_UNITS], 
                                      name = target_var + '_ENC',
                                      return_sequences = self.config[W_ENC][i][W_RSEQ],
                                      return_state = self.config[W_ENC][i][W_RSTATE],
                                      input_shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES])))
            else:
                self.encs.append(LSTM(self.config[W_ENC][i][W_UNITS], 
                                      name = target_var + '_ENC',
                                      return_sequences = self.config[W_ENC][i][W_RSEQ],
                                      return_state = self.config[W_ENC][i][W_RSTATE]))

        self.repeat = RepeatVector(self.config[W_SETTINGS][W_NFUTURE], name = self.target_var + '_REPEAT')

        # Decoder
        self.decs = list()
        for i in range(len(self.config[W_DEC])):
            self.decs.append(LSTM(self.config[W_DEC][i][W_UNITS], 
                                  name = self.target_var + '_DEC',
                                  return_sequences = self.config[W_DEC][i][W_RSEQ]))

        # Time distributed dense
        self.outdense = list()
        for i in range(len(self.config[W_OUT])):
            self.outdense.append(Dense(self.config[W_OUT][i][W_UNITS], activation = self.config[W_OUT][i][W_ACT], 
                                       name = self.target_var + '_D'))
        self.out = Dense(self.config[W_SETTINGS][W_NFUTURE], activation = 'linear', name = self.target_var + '_out')

        # Initialization
        self.past_h = tf.Variable(tf.zeros([self.config[W_ENC][-1][W_UNITS], 1]), 
                                            trainable = False, 
                                            shape = (self.config[W_ENC][-1][W_UNITS], 1),
                                            name = self.target_var + '_pastH')
        self.past_c = tf.Variable(tf.zeros([self.config[W_ENC][-1][W_UNITS], 1]), 
                                            trainable = False, 
                                            shape = (self.config[W_ENC][-1][W_UNITS], 1),
                                            name = self.target_var + '_pastC')


    def call(self, x):
        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            x_tilde = self.ca([x, self.past_h, self.past_c])
            # x_tilde = self.ca(x)

        # Encoder
        enc = x_tilde if self.config[W_SETTINGS][W_USEATT] else x
        for i in range(len(self.config[W_ENC])):
            enc, h, c = self.encs[i](enc)
        self.past_h.assign(tf.expand_dims(h[0], -1))
        self.past_c.assign(tf.expand_dims(c[0], -1))

        repeat = self.repeat(enc)
            
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