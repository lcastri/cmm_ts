import tensorflow as tf
import numpy as np
from keras.layers import *
from models.Constraint import Between
from models.words import *
import keras.backend as K

class SelfAttention(Layer):
    def __init__(self, config, causal_vec : np.array, name = 'Attention'):
        super(SelfAttention, self).__init__(name = name)
        self.config = config
        self.causal_vec = np.tile(causal_vec, (self.config[W_SETTINGS][W_NPAST], 1))
        self.Dg = Dense(self.config[W_INPUTATT][W_UNITS], activation='tanh', use_bias = True)
        
        self.Wa = self.add_weight(name='Wa', shape = (self.config[W_SETTINGS][W_NFEATURES], self.config[W_INPUTATT][W_UNITS]), 
                                  initializer='random_normal', 
                                  trainable = True)

        self.ba = self.add_weight(name='ba', shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]), 
                                  initializer='random_normal', 
                                  trainable = True)
                                  
        if self.config[W_INPUTATT][W_USECAUSAL]:
            constraint = Between(self.causal_vec, self.config[W_INPUTATT][W_TRAINTHRESH]) if self.config[W_INPUTATT][W_CTRAINABLE] and self.config[W_INPUTATT][W_USECONSTRAINT]else None
            self.causal = self.add_weight(name = 'causal', shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]), 
                                        initializer = tf.initializers.Constant(self.causal_vec), 
                                        trainable = self.config[W_INPUTATT][W_CTRAINABLE],
                                        constraint = constraint) 


    def call(self, x):

        # Attention weights
        g = self.Dg(x)
        alpha = tf.matmul(g, self.Wa, transpose_b=True) + self.ba
        if self.config[W_INPUTATT][W_USECAUSAL]: alpha = alpha + self.causal
        alpha = Activation('sigmoid')(alpha)

        # New state
        x_tilde = tf.math.multiply(x, alpha)
        return x_tilde