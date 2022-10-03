import tensorflow as tf
import keras.backend as K
from keras.layers import *
from models.words import *


class InputAttention(Layer):
    def __init__(self, config, name = 'Attention'):
        super(InputAttention, self).__init__(name = name)
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

        super(InputAttention, self).build(input_shape)


    def call(self, inputs):
        x, past_h, past_c = inputs

        # Hidden and cell states concatenation
        conc = K.concatenate([past_h, past_c], axis = 0)
        conc = K.concatenate([conc for _ in range(x.shape[-1])], axis = 1)
        # print("[ht-1, ct-1] shape", conc.shape)

        # Attention weights pre softmax
        e = tf.matmul(tf.transpose(self.Ve), K.tanh(tf.matmul(self.We, conc) + tf.matmul(self.Ue, x)))
        # print("e shape", e.shape)

        # Attention weights
        alpha = tf.nn.softmax(e, axis = 2)
        # print("alpha shape", alpha.shape)

        # New state
        x_tilde = tf.math.multiply(x, alpha)
        # print("x_tilde shape", x_tilde.shape)

        return x_tilde
