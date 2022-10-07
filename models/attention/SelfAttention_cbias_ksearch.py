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
        self.causal_vec = causal_vec
        self.Dg = Dense(self.config["ATTUNITS"], activation = 'tanh', use_bias = True)
        if self.config[W_USECAUSAL]:
            constraint = Between(self.causal_vec, self.config[W_TRAINTHRESH]) if self.config[W_CTRAINABLE] and self.config[W_USECONSTRAINT]else None
            self.Dalpha = Dense(self.config[W_NFEATURES], activation = 'sigmoid',
                                use_bias = True,
                                bias_initializer = tf.initializers.Constant(self.causal_vec),
                                bias_constraint = constraint)
        else:
            self.Dalpha = Dense(self.config[W_NFEATURES], activation = 'sigmoid',
                                use_bias = True)


    def call(self, x):

        # Attention weights
        g = self.Dg(x)
        alpha = self.Dalpha(g)

        # New state
        x_tilde = tf.math.multiply(x, alpha)
        return x_tilde