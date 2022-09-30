import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import *
from models.Constraint import Between
from models.words import *


class CAttention(Layer):
    def __init__(self, config, causal_vec : np.array, name = 'Attention'):
        super(CAttention, self).__init__(name = name)
        self.causal_vec = causal_vec
        self.config = config



    def build(self, x):
        # input_shape = batch x n_past x n_features
        # T = window size : n_past
        # n = number of driving series : n_features
        # m = size of hidden state + cell state
        T = x[1]
        n = x[-1]
        m = self.config[W_ENC][-1][W_UNITS]

        # Wg = m x 1
        self.Wg = self.add_weight(name='Wg', shape=(n, m), 
                                  initializer='random_normal', 
                                  trainable = True)

        self.bg = self.add_weight(name = 'bias_g', shape = (T, m), 
                                  initializer = 'zeros', 
                                  trainable = True)

        # Wa = T x m
        self.Wa = self.add_weight(name = 'Wa', shape = (m, T), 
                                  initializer='random_normal', 
                                  trainable = True)

        self.ba = self.add_weight(name = 'bias_a', shape = (T, n), 
                                  initializer = 'zeros', 
                                  trainable = True)


        if self.config[W_INPUTATT][W_USECAUSAL]:
            constraint = Between(self.causal_vec, self.config[W_INPUTATT][W_TRAINTHRESH]) if self.config[W_INPUTATT][W_CTRAINABLE] and self.config[W_INPUTATT][W_USECONSTRAINT]else None
            self.causal = self.add_weight(name = 'causal', shape = (1, n), 
                            initializer = tf.initializers.Constant(self.causal_vec), 
                            trainable = self.config[W_INPUTATT][W_CTRAINABLE],
                            constraint = constraint) 

        super(CAttention, self).build(x)


    def call(self, x):
        # print("Performing layer", self.name)
        # print("X", x.numpy())
        # print("driving series shape", x.shape)
        # print("ht-1 shape", past_h.shape)
        # print("ct-1 shape", past_h.shape)
        # print("Ve shape", self.Ve.shape)
        # print("We shape", self.We.shape)
        # print("Ue shape", self.Ue.shape)
        # print("bias shape", self.b.shape)
        # print("causal shape", self.causal.shape)

 
        # Attention weights pre softmax
        g = K.tanh(K.dot(x, self.Wg) + self.bg)
        print("g.shape ", g.shape)
        second = K.dot(g, self.Wa)
        print("second.shape ", second.shape)
        alpha = K.sigmoid(K.dot(g, self.Wa) + self.ba)
        if self.config[W_INPUTATT][W_USECAUSAL]: e = e + self.causal
        # print("e shape", e.shape)

        # Attention weights x causal weights
        # e = tf.math.multiply(e, )
        # print("e*causal shape", e.shape)

        # Attention weights
        # print("alpha shape", alpha.shape)

        # New state
        x_tilde = tf.math.multiply(x, alpha)
        # print("x_tilde shape", x_tilde.shape)
        # print("Casual weights", alpha.numpy())
        # print("X tilde", x_tilde.numpy())
        return x_tilde
