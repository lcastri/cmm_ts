from keras.layers import Layer
import keras.initializers 
import tensorflow as tf
import numpy as np

def softmax_stable(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

class CausalAttention(Layer):
    def __init__(self, causal_model, name = 'Causal_Attention'):
        super(CausalAttention,self).__init__(name = name)
        self.causal_model = softmax_stable(causal_model)


    def build(self, input_shape):
        self.W = self.add_weight(name = 'causal_attention_weight', 
                                 shape = (1, input_shape[-1]), 
                                 trainable = True,
                                 initializer = keras.initializers.Constant(self.causal_model))   

        super(CausalAttention, self).build(input_shape)


    def call(self, x):
        x_W = tf.math.multiply(x, self.W)
        return x_W