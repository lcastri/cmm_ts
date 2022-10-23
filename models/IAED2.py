import numpy as np
from constants import CM_FPCMCI
from models.attention.SelfAttention import SelfAttention
from models.attention.InputAttention import InputAttention
from keras.layers import *
from keras.models import *
import tensorflow as tf
import models.Words as W


class IAED(Layer):
    def __init__(self, config, target_var, name = "IAED", searchBest = False):
        super(IAED, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        if self.config[W.USEATT]:

            # Causal vector definition
            if searchBest:
                causal_vec = CM_FPCMCI[0, :] if self.config[W.USECAUSAL] else None
            else:
                causal_vec = np.array(self.config[W.CMATRIX][self.config[W.FEATURES].index(self.target_var), :]) if self.config[W.USECAUSAL] else None
            
            # Self attention
            self.selfatt = SelfAttention(self.config, causal_vec, name = self.target_var + '_selfatt')
            
            # Input attention
            self.inatt = InputAttention(self.config, name = self.target_var + '_inatt')
           
            # Encoders
            self.selfenc1 = LSTM(int(self.config[W.ENCDECUNITS]/2), 
                                name = target_var + '_selfENC1',
                                return_state = True,
                                return_sequences = True,
                                input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
            self.selfenc2 = LSTM(int(self.config[W.ENCDECUNITS]/2), 
                                name = target_var + '_selfENC2',
                                return_state = True,
                                input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))

            self.inenc1 = LSTM(int(self.config[W.ENCDECUNITS]/2),
                              name = target_var + '_inENC1',
                              return_state = True,
                              return_sequences = True,
                              input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
            self.inenc2 = LSTM(int(self.config[W.ENCDECUNITS]/2),
                              name = target_var + '_inENC2',
                              return_state = True,
                              input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))

            # Initialization
            self.past_h = tf.Variable(tf.zeros([int(self.config[W.ENCDECUNITS]/2), 1]), 
                                                trainable = False, 
                                                shape = (int(self.config[W.ENCDECUNITS]/2), 1),
                                                name = self.target_var + '_pastH')
            self.past_c = tf.Variable(tf.zeros([int(self.config[W.ENCDECUNITS]/2), 1]), 
                                                trainable = False, 
                                                shape = (int(self.config[W.ENCDECUNITS]/2), 1),
                                                name = self.target_var + '_pastC')

        else:
            self.enc1 = LSTM(self.config[W.ENCDECUNITS], 
                            name = target_var + '_ENC1',
                            return_state = True,
                            input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
            self.enc2 = LSTM(self.config[W.ENCDECUNITS], 
                            name = target_var + '_ENC2',
                            return_state = True,
                            input_shape = (self.config[W.NPAST], self.config[W.NFEATURES]))

        self.repeat = RepeatVector(self.config[W.NFUTURE], name = self.target_var + '_REPEAT')

        # Decoder
        self.dec = LSTM(self.config[W.ENCDECUNITS], name = self.target_var + '_DEC')

        # Dense
        self.outdense1 = Dense(self.config[W.D1UNITS], activation = self.config[W.D1ACT], name = self.target_var + '_D1')
        self.outdense2 = Dense(self.config[W.D2UNITS], activation = self.config[W.D2ACT], name = self.target_var + '_D2')
        self.outdense3 = Dense(self.config[W.D3UNITS], activation = self.config[W.D3ACT], name = self.target_var + '_D3')
        self.out = Dense(self.config[W.NFUTURE], activation = 'linear', name = self.target_var + '_DOUT')
        

    def call(self, x):
        if self.config[W.USEATT]:
            # Attention
            x_selfatt = self.selfatt(x)
            x_selfatt = Dropout(self.config[W.DRATE])(x_selfatt)

            x_inatt = self.inatt([x, self.past_h, self.past_c])
            x_inatt = Dropout(self.config[W.DRATE])(x_inatt)

            # Encoders
            enc1_1, _, _ = self.selfenc1(x_selfatt)
            enc2_1, _, _ = self.inenc1(x_inatt)
            enc1_2, h1, c1 = self.selfenc2(enc1_1)
            enc2_2, h2, c2 = self.inenc2(enc2_1)
            self.past_h.assign(tf.expand_dims(h2[-1], -1))
            self.past_c.assign(tf.expand_dims(c2[-1], -1))

            x = concatenate([enc1_2, enc2_2])
            if self.config[W.DECINIT]:
                h = concatenate([h1, h2])
                c = concatenate([c1, c2])
        else:
            x, h, c = self.enc(x)
            
        repeat = self.repeat(x)
            
        # Decoder
        if self.config[W.DECINIT]:
            dec = self.dec(repeat, initial_state = [h, c])
        else:
            dec = self.dec(repeat)

        y = Dropout(self.config[W.DRATE])(dec)
        y = self.outdense1(dec)
        y = Dropout(self.config[W.DRATE])(y)
        y = self.outdense2(y)
        y = Dropout(self.config[W.DRATE])(y)
        y = self.outdense3(y)
        y = Dropout(self.config[W.DRATE])(y)
        y = self.out(y)
        y = tf.expand_dims(y, axis = -1)

        return y