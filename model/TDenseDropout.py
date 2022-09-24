from keras.layers import *
from keras.models import *


class TDenseDropout(Layer):
    def __init__(self, units, activation, name = "TDD", dropout = None):
        super(TDenseDropout, self).__init__(name = name)

        # Time distributed dense
        self.dense = TimeDistributed(Dense(units, activation = activation))
        self.dropout = Dropout(dropout) if dropout is not None else None


    def call(self, x):
        y = self.dense(x)
        if self.dropout is not None: y = self.dropout(y)

        return y


    
              
