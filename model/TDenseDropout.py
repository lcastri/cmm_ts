from keras.layers import *
from keras.models import *


class TDenseDropout(Layer):
    def __init__(self, units, dropout, activation, name = "TDD"):
        super(TDenseDropout, self).__init__(name = name)

        # Time distributed dense
        self.dense = TimeDistributed(Dense(units, activation = activation))
        self.dropout = Dropout(dropout)


    def call(self, x):
        y = self.dense(x)
        y = self.dropout(y)

        return y


    
              
