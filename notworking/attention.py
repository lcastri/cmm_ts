from keras.layers import Layer
import keras.initializers 
import tensorflow as tf
from Constraint import Between

W_TRAINABLE = 'trainable'
W_ADJTHRES = 'constraint_threshold'

class CausalAttention(Layer):
    def __init__(self, causal_model, name = 'Causal_Attention', config = None):
        super(CausalAttention,self).__init__(name = name)
        self.causal_model = causal_model
        self.trainable = config[W_TRAINABLE] if W_TRAINABLE in config.keys() else True
        self.adj_thres = config[W_ADJTHRES] if W_ADJTHRES in config.keys() else 0.1
        self.W = list()


    def build(self, input_shape):
        for cw_idx in range(len(self.causal_model)):
            self.W.append(self.add_weight(name = str(cw_idx) + '_ca_weight',  
                                          trainable = self.trainable,
                                          initializer = keras.initializers.Constant(self.causal_model[cw_idx]),
                                          constraint = Between(self.causal_model[cw_idx], self.adj_thres) if self.trainable else None))
        # if not self.trainable: self.weight_softmax()
        super(CausalAttention, self).build(input_shape)


    def weight_softmax(self):
        # Creating partition based on condition:
        condition_mask = tf.cast(tf.greater(self.W, 0.), tf.int32)
        partitioned_T = tf.dynamic_partition(self.W, condition_mask, 2)
        # Applying the operation to the target partition:
        partitioned_T[1] = tf.nn.softmax(partitioned_T[1])

        # Stitching back together, flattening T and its indices to make things easier::
        condition_indices = tf.dynamic_partition(tf.range(tf.size(self.W)), tf.reshape(condition_mask, [-1]), 2)
        res_mask = tf.dynamic_stitch(condition_indices, partitioned_T)
        res_mask = tf.reshape(res_mask, tf.shape(self.W))

        for cw_idx in range(len(self.causal_model)): self.W[cw_idx].assign(res_mask[cw_idx])
        return res_mask


    def call(self, x):
        # if self.trainable: self.weight_softmax()  
        x_W = tf.math.multiply(x, self.W)
        return x_W
        # x_W = tf.math.multiply(x, self.W)
        # return x_W


    def get_config(self):
        cw = dict()
        for w in self.W:
            cw[w.name] = w.numpy()
        return {"causal_attention_weights": cw}


    @classmethod
    def from_config(cls, config):
        return cls(**config)