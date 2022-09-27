from abc import ABC, abstractmethod
import os
import models.utils as utils
from models.words import *
from keras.models import *
from matplotlib import pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
from math import sqrt
from sklearn.metrics import mean_squared_error


class MyModel(ABC):
    def __init__(self, config):
        self.model_dir, self.plot_dir, self.pred_dir = utils.create_dir(config[W_SETTINGS][W_FOLDER])
        utils.no_warning()
        self.config = config
        self.model : Model = None


    @abstractmethod
    def create_model(self) -> Model:
        pass


    def fit(self, X, y, validation_data, batch_size, epochs, callbacks = None):
        history = self.model.fit(x = X, y = y, batch_size = batch_size, epochs = epochs,
                                 callbacks = callbacks, validation_data = validation_data)
            
        if "loss" in history.history.keys():
            plt.figure()
            plt.plot(history.history["loss"], label = "Training loss")
            plt.plot(history.history["val_loss"], label = "Validation loss")
            plt.legend()
            plt.grid()
            plt.savefig(self.plot_dir + "/loss.png", dpi = 300)
            plt.savefig(self.plot_dir + "/loss.eps", dpi = 300)
            plt.close()

        if "mae" in history.history.keys():
            plt.figure()
            plt.plot(history.history["mae"], label = "Training mae")
            plt.plot(history.history["val_mae"], label = "Validation mae")
            plt.legend()
            plt.grid()
            plt.savefig(self.plot_dir + "/mae.png", dpi = 300)
            plt.savefig(self.plot_dir + "/mae.eps", dpi = 300)
            plt.close()

        if "mape" in history.history.keys():
            plt.figure()
            plt.plot(history.history["mape"], label = "Training mape")
            plt.plot(history.history["val_mape"], label = "Validation mape")
            plt.legend()
            plt.grid()
            plt.savefig(self.plot_dir + "/mape.png", dpi = 300)
            plt.savefig(self.plot_dir + "/mape.eps", dpi = 300)
            plt.close()

        if "accuracy" in history.history.keys():
            plt.figure()
            plt.plot(history.history["accuracy"], label = "Training accuracy")
            plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
            plt.legend()
            plt.grid()
            plt.savefig(self.plot_dir + "/accuracy.png", dpi = 300)
            plt.savefig(self.plot_dir + "/accuracy.eps", dpi = 300)
            plt.close()

        with open(self.model_dir + '/history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)



    def RMSE(self, X, y, scaler, show = False):
        print('\n##')
        print('## Prediction evaluation through RMSE')
        print('##')

        predY = self.model.predict(X)
        rmse = np.zeros(shape = (1, y.shape[1]))
        for t in tqdm(range(len(y)), desc = 'RMSE'):
            actualY_t = np.squeeze(y[t,:,:])
            predY_t = np.squeeze(predY[t,:,:])
            actualY_t = scaler.inverse_transform(actualY_t)
            predY_t = scaler.inverse_transform(predY_t)
            rmse = rmse + np.array([sqrt(mean_squared_error(actualY_t[f], predY_t[f])) for f in range(self.config[W_SETTINGS][W_NFUTURE])])
        rmse_mean = np.sum(rmse, axis=0)/len(y)

        plt.figure()
        plt.title("Mean RMSE vs time steps")
        plt.plot(range(self.config[W_SETTINGS][W_NFUTURE]), rmse_mean)
        plt.xlabel("Time steps")
        plt.xlabel("Mean RMSE")
        plt.grid()
        if show:
            plt.show()
        else:
            plt.savefig(self.plot_dir + "/rmse_pred.png", dpi = 300)
            plt.savefig(self.plot_dir + "/rmse_pred.eps", dpi = 300)
        plt.close()
        return rmse_mean


    def plot_predictions(self, X, y, scaler):
        print('\n##')
        print('## Predictions')
        print('##')

        predY = self.model.predict(X)
        for f in self.config[W_SETTINGS][W_FEATURES]:

            # Create var folder
            if not os.path.exists(self.pred_dir + "/" + str(self.target_var) + "/"):
                os.makedirs(self.pred_dir + "/" + str(self.target_var) + "/")

            f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(f)

            for t in tqdm(range(len(predY)), desc = f):
                # test X
                X_t = np.squeeze(X[t,:,:])
                X_t = scaler.inverse_transform(X_t)

                # test y
                Y_t = np.squeeze(y[t,:,:])
                Y_t = scaler.inverse_transform(Y_t)

                # pred y
                predY_t = np.squeeze(predY[t,:,:])
                predY_t = scaler.inverse_transform(predY_t)

                plt.plot(range(t, t + len(X_t[:, f_idx])), X_t[:, f_idx], color = 'green', label = "past")
                plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(Y_t[:, f_idx])), Y_t[:, f_idx], color = 'blue', label = "actual")
                plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(predY_t[:, f_idx])), predY_t[:, f_idx], color = 'red', label = "pred")
                plt.title("Multi-step prediction - " + f)
                plt.xlabel("step = 0.1s")
                plt.ylabel(f)
                plt.grid()
                plt.legend()
                plt.savefig(self.pred_dir + "/" + str(f) + "/" + str(t) + ".png")

                plt.clf()
                
        plt.close()