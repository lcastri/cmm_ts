import tensorflow as tf
import numpy as np
from keras.layers import *
from models.Constraint import Between
from models.words import *


class SelfAttention(Layer):
    def __init__(self, config, causal_vec : np.array, name = 'Attention'):
        super(SelfAttention, self).__init__(name = name)
        self.causal_vec = causal_vec
        self.config = config
        self.Dg = Dense(self.config[W_INPUTATT][W_UNITS], activation='tanh', use_bias = True)
        self.Dalpha = Dense(self.config[W_SETTINGS][W_NFEATURES], activation='sigmoid', use_bias = True) 
        
        if self.config[W_INPUTATT][W_USECAUSAL]:
            constraint = Between(self.causal_vec, self.config[W_INPUTATT][W_TRAINTHRESH]) if self.config[W_INPUTATT][W_CTRAINABLE] and self.config[W_INPUTATT][W_USECONSTRAINT]else None
            self.causal = self.add_weight(name = 'causal', shape = (1, self.config[W_SETTINGS][W_NFEATURES]), 
                                          initializer = tf.initializers.Constant(self.causal_vec), 
                                          trainable = self.config[W_INPUTATT][W_CTRAINABLE],
                                          constraint = constraint) 


    def call(self, x):

        # Attention weights
        g = self.Dg(x)
        alpha = self.Dalpha(g)
        if self.config[W_INPUTATT][W_USECAUSAL]: alpha = Activation('softmax')(alpha + self.causal)

        # New state
        x_tilde = tf.math.multiply(x, alpha)
        return x_tilde
