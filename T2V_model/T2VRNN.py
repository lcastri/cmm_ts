from keras.layers import *
from .T2V import T2V
from words import *
import numpy as np
import tensorflow as tf
from .Attention import CAttention


class T2VRNN(Layer):
    
    def __init__(self, config, target_var, name):
        super(T2VRNN, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            self.ca = CAttention(self.config, 
                                 np.array(self.config[W_INPUTATT][W_CMATRIX][self.config[W_SETTINGS][W_FEATURES].index(self.target_var), :]), 
                                 name = self.target_var + '_CA')
           
        # T2V
        self.t2v = T2V(config[W_T2V][W_UNITS])

        # RNN block
        self.rnn = LSTM(self.config[W_RNN][W_UNITS], activation = 'tanh', name = self.target_var + '_lstm')

        # Dense
        self.outdense = list()
        for i in range(len(self.config[W_OUT])):
            self.outdense.append(Dense(self.config[W_OUT][i][W_UNITS], 
                                       activation = self.config[W_OUT][i][W_ACT], 
                                       name = self.target_var + '_dense'))
        self.out = Dense(self.config[W_SETTINGS][W_NFUTURE], activation='linear', name = self.target_var + '_out')

        # # Initialization
        # self.past_h = tf.Variable(tf.zeros([self.config[W_ENC][-1][W_UNITS], 1]), trainable = False, 
        #                                     shape = (self.config[W_ENC][-1][W_UNITS], 1))
        # self.past_c = tf.Variable(tf.zeros([self.config[W_ENC][-1][W_UNITS], 1]), trainable = False, 
        #                                     shape = (self.config[W_ENC][-1][W_UNITS], 1))

        
    def call(self, x):
        y = self.t2v(x)
        y = self.rnn(y)
        for i in range(len(self.config[W_OUT])): y = self.outdense[i](y)
        y = self.out(y)
        return tf.expand_dims(y, -1)