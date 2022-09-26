import numpy as np
from .Attention import CAttention
from keras.layers import *
from keras.models import *
import tensorflow as tf
from .TDenseDropout import TDenseDropout
from ..words import *
from matplotlib import pyplot as plt
import pickle


class IAED(Model):
    def __init__(self, config, target_var, name = "IAED"):
        super(IAED, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            self.ca = CAttention(self.config, 
                                 np.array(self.config[W_INPUTATT][W_CMATRIX][self.config[W_SETTINGS][W_FEATURES].index(self.target_var), :]), 
                                 name = self.target_var + '_CA')
           
        # Encoders
        self.encs = list()
        for i in range(len(self.config[W_ENC])):
            if i == 0:
                self.encs.append(LSTM(self.config[W_ENC][i][W_UNITS], 
                                      name = target_var + '_ENC',
                                      return_sequences = self.config[W_ENC][i][W_RSEQ],
                                      return_state = self.config[W_ENC][i][W_RSTATE],
                                      input_shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES])))
            else:
                self.encs.append(LSTM(self.config[W_ENC][i][W_UNITS], 
                                      name = target_var + '_ENC',
                                      return_sequences = self.config[W_ENC][i][W_RSEQ],
                                      return_state = self.config[W_ENC][i][W_RSTATE]))

        self.repeat = RepeatVector(self.config[W_SETTINGS][W_NFUTURE], name = self.target_var + '_REPEAT')

        # Decoder
        self.decs = list()
        for i in range(len(self.config[W_DEC])):
            self.decs.append(LSTM(self.config[W_DEC][i][W_UNITS], 
                                  name = self.target_var + '_DEC',
                                  return_sequences = self.config[W_DEC][i][W_RSEQ]))

        # Time distributed dense
        self.outdense = list()
        for i in range(len(self.config[W_OUT])):
            self.outdense.append(TDenseDropout(self.config[W_OUT][i][W_UNITS], 
                                               dropout = self.config[W_OUT][i][W_DROPOUT],
                                               activation = self.config[W_OUT][i][W_ACT], 
                                               name = self.target_var + '_TDD'))
        self.out = TimeDistributed(Dense(1, activation='linear'), name = self.target_var + '_out')

        # Initialization
        self.past_h = tf.Variable(tf.zeros([self.config[W_ENC][-1][W_UNITS], 1]), 
                                            trainable = False, 
                                            shape = (self.config[W_ENC][-1][W_UNITS], 1))
        self.past_c = tf.Variable(tf.zeros([self.config[W_ENC][-1][W_UNITS], 1]), 
                                            trainable = False, 
                                            shape = (self.config[W_ENC][-1][W_UNITS], 1))


    def call(self, x):
        # Input attention
        if self.config[W_SETTINGS][W_USEATT]:
            x_tilde = self.ca([x, self.past_h, self.past_c])

        # Encoder
        enc = x_tilde if self.config[W_SETTINGS][W_USEATT] else x
        for i in range(len(self.config[W_ENC])):
            enc, h, c = self.encs[i](enc)
        self.past_h.assign(tf.expand_dims(h[0], -1))
        self.past_c.assign(tf.expand_dims(c[0], -1))

        repeat = self.repeat(enc)
            
        # Decoder
        dec = repeat
        for i in range(len(self.config[W_DEC])):
            dec = self.decs[i](dec)

        y = dec
        for i in range(len(self.config[W_OUT])):
            y = self.outdense[i](y)
        y = self.out(y)

        return y

    def model(self):
        x = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
        return Model(inputs = [x], outputs = self.call(x))


    def get_config(self):
        data = dict()
        if self.config[W_SETTINGS][W_USEATT] and self.config[W_INPUTATT][W_USECAUSAL]:
            for var in self.config[W_SETTINGS][W_FEATURES]:
                data[var] = self.ca.causal.numpy()
        return {"causal_weights": data}


    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        history = super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
        
        plt.figure()
        plt.plot(history.history["loss"], label = "Training loss")
        plt.plot(history.history["val_loss"], label = "Validation loss")
        plt.legend()
        plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/loss.png", dpi = 300)
        plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/loss.eps", dpi = 300)

        plt.figure()
        plt.plot(history.history["mae"], label = "Training mae")
        plt.plot(history.history["val_mae"], label = "Validation mae")
        plt.legend()
        plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/mae.png", dpi = 300)
        plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/mae.eps", dpi = 300)

        plt.figure()
        plt.plot(history.history["mape"], label = "Training mape")
        plt.plot(history.history["val_mape"], label = "Validation mape")
        plt.legend()
        plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/mape.png", dpi = 300)
        plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/mape.eps", dpi = 300)

        plt.figure()
        plt.plot(history.history["accuracy"], label = "Training accuracy")
        plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
        plt.legend()
        plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/accuracy.png", dpi = 300)
        plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/accuracy.eps", dpi = 300)

        with open(self.config[W_SETTINGS][W_FOLDER] + '/history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    
              
