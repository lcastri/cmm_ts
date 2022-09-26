import pickle
from keras.layers import *
from keras.models import *
from .T2VRNN import T2VRNN
from matplotlib import pyplot as plt
from models.words import *
import models.utils as utils
from tqdm import tqdm
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import os

class mT2VRNN(Model):
    def __init__(self, config):
        super(mT2VRNN, self).__init__()
        self.config = config
        self.channels = dict()

        utils.create_folder(self.config[W_SETTINGS][W_FOLDER])
        utils.no_warning()

        # Multihead
        for var in self.config[W_SETTINGS][W_FEATURES]:
            self.channels[var] = T2VRNN(self.config, var, name = var + "_T2VRNN")

        # Concatenation
        self.concat = Concatenate(axis = 2)
    

    def call(self, input_sequence):
        resMultiHead = list()
        x = input_sequence

        for var in self.config[W_SETTINGS][W_FEATURES]:
            resMultiHead.append(self.channels[var](x))
        
        concat = self.concat(resMultiHead)

        return concat

    
    def model(self):
        x = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
        return Model(inputs = [x], outputs = self.call(x))


    def get_config(self):
        data = dict()
        if self.config[W_SETTINGS][W_USEATT] and self.config[W_INPUTATT][W_USECAUSAL]:
            for var in self.config[W_SETTINGS][W_FEATURES]:
                data[var] = self.channels[var].ca.causal.numpy()
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
            actualY_t = np.squeeze(y[t,:,:])
            predY_t = np.squeeze(predY[t,:,:])
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
        for f in self.config[W_SETTINGS][W_FEATURES]:

            # Create var folder
            if not os.path.exists(self.config[W_SETTINGS][W_FOLDER] + "/predictions/" + str(f) + "/"):
                os.makedirs(self.config[W_SETTINGS][W_FOLDER] + "/predictions/" + str(f) + "/")

            f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(f)

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
                plt.title("Multi-step prediction - " + f)
                plt.xlabel("step = 0.1s")
                plt.ylabel(f)
                plt.legend()
                plt.savefig(self.config[W_SETTINGS][W_FOLDER] + "/predictions/" + str(f) + "/" + str(t) + ".png")

                plt.clf()
                
        plt.close()