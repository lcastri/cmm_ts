import numpy as np
from Attention import CAttention
from keras.layers import *
from keras.models import *
import tensorflow as tf
from TDenseDropout import TDenseDropout

class IAED(Layer):
    def __init__(self, enc_dec_units, out_units, n_past, n_feature, n_future, 
                 target_var, use_attention, use_causality, causal_vec,  name = "IAED"):
        super(IAED, self).__init__(name = name)
        self.enc_dec_units = enc_dec_units
        self.out_units = out_units
        self.n_past = n_past
        self.n_feature = n_feature
        self.n_future = n_future
        self.target_var = target_var

        # Input attention
        self.use_attention = use_attention

        if self.use_attention:
            self.ca = CAttention(self.enc_dec_units, use_causality, np.array(causal_vec), 
                                 name = self.target_var + '_CA')
           
        # Encoder
        self.enc_lstm = LSTM(self.enc_dec_units, 
                             name = target_var + '_ENC',
                             return_sequences = False,
                             return_state = True,
                             input_shape = (self.n_past, self.n_feature))

        self.repeat = RepeatVector(self.n_future, name = self.target_var + '_REPEAT')

        # Decoder
        self.dec_lstm = LSTM(self.enc_dec_units, 
                             name = self.target_var + '_DEC',
                             return_sequences = True)

        # Time distributed dense
        self.outdense = list()
        for u in self.out_units:
            self.outdense.append(TDenseDropout(u, 0.5, 'relu', name = self.target_var + '_TDD'))
        self.out = TimeDistributed(Dense(1, activation='linear'), name = self.target_var + '_out')

        # Initialization
        self.past_h = tf.Variable(tf.zeros([self.enc_dec_units, 1]), trainable = False, shape = (self.enc_dec_units, 1))
        self.past_c = tf.Variable(tf.zeros([self.enc_dec_units, 1]), trainable = False, shape = (self.enc_dec_units, 1))


    def call(self, x):

        # Input attention
        if self.use_attention:
            x_tilde = self.ca([x, self.past_h, self.past_c])

        # Encoder
        enclstm, h, c = self.enc_lstm(x_tilde if self.use_attention else x)
        self.past_h.assign(tf.expand_dims(h[0], -1))
        self.past_c.assign(tf.expand_dims(c[0], -1))

        repeat = self.repeat(enclstm)
            
        # Decoder
        declstm = self.dec_lstm(repeat)

        y = declstm
        for u in range(len(self.out_units)):
            y = self.outdense[u](y)
        y = self.out(y)

        return y


    
              
