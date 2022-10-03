import tensorflow as tf
import numpy as np
from keras.layers import *
from models.Constraint import Between
from models.words import *


class CAttention(Layer):
    def __init__(self, config, causal_vec : np.array, name = 'Attention'):
        super(CAttention, self).__init__(name = name)
        self.causal_vec = causal_vec
        self.config = config


    def build(self, input_shape):
        n = input_shape[-1]
        constraint = Between(self.causal_vec, self.config[W_INPUTATT][W_TRAINTHRESH]) if self.config[W_INPUTATT][W_CTRAINABLE] and self.config[W_INPUTATT][W_USECONSTRAINT] else None
        self.causal = self.add_weight(name = 'causal', shape = (1, n), 
                                      initializer = tf.initializers.Constant(self.causal_vec), 
                                      trainable = self.config[W_INPUTATT][W_CTRAINABLE],
                                      constraint = constraint) 

        super(CAttention, self).build(input_shape)


    def call(self, x):
        alpha = tf.nn.softmax(self.causal, axis = 1)
        x_tilde = tf.math.multiply(x, alpha)
        return x_tilde

    
    def get_config(self):
        return {"causal_vector": self.causal}