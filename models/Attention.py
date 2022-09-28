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


    def build(self, inputs):
        # input_shape = batch x n_past x n_features
        # T = window size : n_past
        # n = number of driving series : n_features
        # m = size of hidden state + cell state
        input_shape = inputs[0]
        T = input_shape[1]
        n = input_shape[-1]
        m = inputs[1][0] + inputs[2][0]
        # m = self.units

        # Ve = T x 1
        self.Ve = self.add_weight(name='Ve', shape=(T, 1), 
                                  initializer='random_normal', 
                                  trainable = True)

        # We = T x 2m
        self.We = self.add_weight(name = 'We', shape = (T, m), 
                                  initializer='random_normal', 
                                  trainable = True)

        # Ue = T x T
        self.Ue = self.add_weight(name = 'Ue', shape = (T, T), 
                                  initializer = 'random_normal', 
                                  trainable = True)

        self.b = self.add_weight(name = 'bias', shape = (T, n), 
                                 initializer = 'zeros', 
                                 trainable = True)

        if self.config[W_INPUTATT][W_USECAUSAL]:
            print("CIAOOO")
            constraint = Between(self.causal_vec, self.config[W_INPUTATT][W_TRAINTHRESH]) if self.config[W_INPUTATT][W_CTRAINABLE] and self.config[W_INPUTATT][W_USECONSTRAINT]else None
            self.causal = self.add_weight(name = 'causal', shape = (1, n), 
                            initializer = tf.initializers.Constant(self.causal_vec), 
                            trainable = self.config[W_INPUTATT][W_CTRAINABLE],
                            constraint = constraint) 

        super(CAttention, self).build(input_shape)


    def call(self, inputs):
        x, past_h, past_c = inputs
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

        # Hidden and cell states concatenation
        conc = K.concatenate([past_h, past_c], axis = 0)
        conc = K.concatenate([conc for _ in range(x.shape[-1])], axis = 1)
        # print("[ht-1, ct-1] shape", conc.shape)

        # Attention weights pre softmax
        e = tf.matmul(tf.transpose(self.Ve), K.tanh(tf.matmul(self.We, conc) + tf.matmul(self.Ue, x) + self.b))
        if self.config[W_INPUTATT][W_USECAUSAL]: e = e + self.causal
        # print("e shape", e.shape)

        # Attention weights x causal weights
        # e = tf.math.multiply(e, )
        # print("e*causal shape", e.shape)

        # Attention weights
        alpha = tf.nn.softmax(e, axis = 2)
        # print("alpha shape", alpha.shape)

        # New state
        x_tilde = tf.math.multiply(x, alpha)
        # print("x_tilde shape", x_tilde.shape)
        # print("Casual weights", alpha.numpy())
        # print("X tilde", x_tilde.numpy())
        return x_tilde
