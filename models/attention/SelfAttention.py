import tensorflow as tf
import numpy as np
from keras.layers import *
from models.Constraints import *
import models.Words as W
import keras.backend as K

class SelfAttention(Layer):
    def __init__(self, config, causal_vec : np.array, name = 'Attention'):
        super(SelfAttention, self).__init__(name = name)
        self.config = config
        self.causal_vec = causal_vec
        self.Dg = Dense(self.config[W.ATTUNITS], activation = 'tanh', use_bias = True)
        if self.config[W.USECAUSAL]:
            if not self.config[W.CTRAINABLE]:
                constraint = Constant(self.causal_vec)
            elif self.config[W.CTRAINABLE] and not self.config[W.USECONSTRAINT]:
                constraint = None
            elif self.config[W.CTRAINABLE] and self.config[W.USECONSTRAINT]:
                constraint = Between(self.causal_vec, self.config[W.TRAINTHRESH])
            self.Dalpha = Dense(self.config[W.NFEATURES], activation = 'sigmoid',
                                use_bias = True,
                                bias_initializer = tf.initializers.Constant(self.causal_vec),
                                bias_constraint = constraint)
        else:
            self.Dalpha = Dense(self.config[W.NFEATURES], activation = 'sigmoid',
                                use_bias = True)


    def call(self, x):

        # Attention weights
        g = self.Dg(x)
        alpha = self.Dalpha(g)

        # New state
        x_tilde = tf.math.multiply(x, alpha)
        return x_tilde