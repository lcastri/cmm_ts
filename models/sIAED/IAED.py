import numpy as np
from models.Attention import CAttention
from keras.layers import *
from keras.models import *
import tensorflow as tf
from models.words import *
from matplotlib import pyplot as plt
import pickle
import models.utils as utils
from tqdm import tqdm
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import os


class IAED(Model):
    def __init__(self, config, target_var, name = "IAED"):
        super(IAED, self).__init__(name = name)
        self.config = config
        self.target_var = target_var

        utils.create_folder(self.config[W_SETTINGS][W_FOLDER])
        utils.no_warning()

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

        # Dense
        self.outdense = list()
        for i in range(len(self.config[W_OUT])):
            self.outdense.append(Dense(self.config[W_OUT][i][W_UNITS], activation = self.config[W_OUT][i][W_ACT], 
                                       name = self.target_var + '_D'))
        self.out = Dense(self.config[W_SETTINGS][W_NFUTURE], activation = 'linear', name = self.target_var + '_out')

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

    
    def RMSE(self, X, y, scalerOUT, show = False):
        predY = self.predict(X)
        rmse = np.zeros(shape = (1, y.shape[1]))
        for t in tqdm(range(len(y))):
            actualY_t = np.reshape(np.squeeze(y[t,:,:]), newshape = (len(np.squeeze(y[t,:,:])), 1))
            predY_t = np.reshape(np.squeeze(predY[t,:]), newshape = (len(np.squeeze(predY[t,:])), 1))
            actualY_t = scalerOUT.inverse_transform(actualY_t)
            predY_t = scalerOUT.inverse_transform(predY_t)
            rmse = rmse + np.array([sqrt(mean_squared_error(actualY_t[f], predY_t[f])) for f in range(self.config[W_SETTINGS][W_NFUTURE])])
        rmse_mean = np.sum(rmse, axis=0)/len(y)

        plt.figure()
        plt.title("Mean RMSE vs time steps")
        plt.plot(range(self.config[W_SETTINGS][W_NFUTURE]), rmse_mean)
        plt.xlabel("Time steps")
        plt.xlabel("Mean RMSE")
        if show:
            plt.show()
        else:
            plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/rmse_pred.png", dpi = 300)
            plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/plots/rmse_pred.eps", dpi = 300)
        return rmse_mean
        

    def plot_predictions(self, X, y, scalerIN, scalerOUT):

        # Create prediction folder
        if not os.path.exists(self.config[W_SETTINGS][W_FOLDER] + "/predictions/"):
            os.makedirs(self.config[W_SETTINGS][W_FOLDER] + "/predictions/")

        predY = self.predict(X)

        # Create var folder
        if not os.path.exists(self.config[W_SETTINGS][W_FOLDER] + "/predictions/" + str(self.target_var) + "/"):
            os.makedirs(self.config[W_SETTINGS][W_FOLDER] + "/predictions/" + str(self.target_var) + "/")

        f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(self.target_var)

        for t in tqdm(range(len(predY))):
            # test X
            X_t = np.squeeze(X[t,:,:])
            X_t = scalerIN.inverse_transform(X_t)

            # test y
            Y_t = np.squeeze(y[t,:,:])
            Y_t = scalerOUT.inverse_transform(Y_t)

            # pred y
            predY_t = np.squeeze(predY[t,:,:])
            predY_t = scalerOUT.inverse_transform(predY_t)

            plt.plot(range(t, t + len(X_t[:, f_idx])), X_t[:, f_idx], color = 'green', label = "past")
            plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(Y_t[:, f_idx])), Y_t[:, f_idx], color = 'blue', label = "actual")
            plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(predY_t[:, f_idx])), predY_t[:, f_idx], color = 'red', label = "pred")
            plt.title("Multi-step prediction - " + self.target_var)
            plt.xlabel("step = 0.1s")
            plt.ylabel(self.target_var)
            plt.legend()
            plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/predictions/" + str(self.target_var) + "/" + str(t) + ".png")

            plt.clf()
                
        plt.close()