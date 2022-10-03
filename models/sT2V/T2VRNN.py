from keras.layers import *
from .T2V import T2V
from models.words import *
import numpy as np
import tensorflow as tf
from models.attention.InputAttention import InputAttention


class T2VRNN(Layer):
    
    def __init__(self, config, target_var, name = "T2VRNN"):
        super(T2VRNN, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            self.ca = InputAttention(self.config, 
                                 np.array(self.config[W_INPUTATT][W_CMATRIX][self.config[W_SETTINGS][W_FEATURES].index(self.target_var), :]), 
                                 name = self.target_var + '_CA')
           
        # T2V
        self.t2v = T2V(config[W_T2V][W_UNITS], name = self.target_var + '_t2v')

        # RNN block
        self.rnn = LSTM(self.config[W_RNN][W_UNITS],
                        activation = 'tanh',
                        return_state = self.config[W_SETTINGS][W_USEATT],
                        name = self.target_var + '_lstm')

        # Dense
        self.outdense = list()
        for i in range(len(self.config[W_OUT])):
            self.outdense.append(Dense(self.config[W_OUT][i][W_UNITS], 
                                       activation = self.config[W_OUT][i][W_ACT], 
                                       name = self.target_var + '_dense'))
        self.out = Dense(self.config[W_SETTINGS][W_NFUTURE], activation = 'linear', name = self.target_var + '_out')

        # Initialization
        self.past_h = tf.Variable(tf.zeros([self.config[W_RNN][W_UNITS], 1]), trainable = False, 
                                            shape = (self.config[W_RNN][W_UNITS], 1))
        self.past_c = tf.Variable(tf.zeros([self.config[W_RNN][W_UNITS], 1]), trainable = False, 
                                            shape = (self.config[W_RNN][W_UNITS], 1))

        
    def call(self, x):
        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            x_tilde = self.ca([x, self.past_h, self.past_c])

        y = self.t2v(x_tilde if self.config[W_SETTINGS][W_USEATT] else x)
        if self.config[W_SETTINGS][W_USEATT]:
            y, h, c = self.rnn(y)
            self.past_h.assign(tf.expand_dims(h[0], -1))
            self.past_c.assign(tf.expand_dims(c[0], -1))
        else:
            y = self.rnn(y)        
        for i in range(len(self.config[W_OUT])): y = self.outdense[i](y)
        y = self.out(y)
        # return y
        return tf.expand_dims(y, -1)


    def get_config(self):
        data = dict()
        if self.config[W_SETTINGS][W_USEATT] and self.config[W_INPUTATT][W_USECAUSAL]:
            data = self.ca.causal.numpy()
        return {self.target_var + "_causal_weights": data}