from keras.layers import *
from keras.models import *


class DenseDropout(Layer):
    def __init__(self, units, activation, dropout):
        super(DenseDropout, self).__init__()
        self.dbit = dropout != 0
        self.dense = Dense(units, activation = activation)
        if self.dbit: self.dropout = Dropout(dropout)



    def call(self, x):
        y = self.dense(x)
        if self.dbit: y = self.dropout(y)

        return y