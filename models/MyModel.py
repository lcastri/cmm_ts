from abc import ABC, abstractmethod
import os
from constants import RESULT_DIR
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
    def __init__(self, config : dict = None, folder : str = None):
        """
        Constructur, specify config if you want to create a new model, while, set folder if you want to load a pre-existing model

        Args:
            config (dict): configuration file. Default None.
            folder (str): model's name to load. Default None.
        """
        if config:
            self.dir = config[W_SETTINGS][W_FOLDER]
            with open(self.model_dir + '/config.pkl', 'wb') as file_pi:
                pickle.dump(config, file_pi)

            utils.no_warning()
            self.config = config
            self.model : Model = None
        
        if folder:
            self.dir = folder
            with open(self.model_dir + '/config.pkl', 'rb') as pickle_file:
                self.config = pickle.load(pickle_file)
            self.model : Model = load_model(self.model_dir)


    @property
    def model_dir(self):
        model_dir = RESULT_DIR + "/" + self.dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir


    @property
    def plot_dir(self):
        plot_dir = self.model_dir + "/plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        return plot_dir


    @property
    def pred_dir(self):
        pred_dir = self.model_dir + "/predictions"
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        return pred_dir


    @abstractmethod
    def create_model(self) -> Model:
        pass


    def fit(self, X, y, validation_data, batch_size, epochs, callbacks = None):
        """
        Fit wrapper

        Args:
            X (array): X training set
            y (array): Y training set
            validation_data (tuple): (x_val, y_val)
            batch_size (int): batch size
            epochs (int): # epochs
            callbacks (list, optional): List of callbacks. Defaults to None.
        """
        history = self.model.fit(x = X, y = y, batch_size = batch_size, epochs = epochs,
                                 callbacks = callbacks, validation_data = validation_data)
            
        with open(self.model_dir + '/history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        self.plot_history(history)


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
        rmse_mean = np.sum(rmse, axis = 0)/len(y)

        with open(self.model_dir + '/rmse.npy', 'wb') as file:
            np.save(file, rmse_mean)

        self.plot_RMSE(rmse_mean, show = show)
        return rmse


    def predict(self, X, y, scaler):
        print('\n##')
        print('## Predictions')
        print('##')
        x_npy = list()
        ya_npy = list()
        yp_npy = list()

        # Generate and save predictions
        predY = self.model.predict(X)
        for t in range(len(predY)):
            # test X
            X_t = np.squeeze(X[t,:,:])
            X_t = scaler.inverse_transform(X_t)
            x_npy.append(X_t)

            # test y
            Y_t = np.squeeze(y[t,:,:])
            Y_t = scaler.inverse_transform(Y_t)
            ya_npy.append(Y_t)

            # pred y
            predY_t = np.squeeze(predY[t,:,:])
            predY_t = scaler.inverse_transform(predY_t)
            yp_npy.append(predY_t)
            
        with open(self.pred_dir + '/x_npy.npy', 'wb') as file:
            np.save(file, x_npy)
        with open(self.pred_dir + '/ya_npy.npy', 'wb') as file:
            np.save(file, ya_npy)
        with open(self.pred_dir + '/yp_npy.npy', 'wb') as file:
            np.save(file, yp_npy)

        self.plot_prediction(x_npy, ya_npy, yp_npy)


        # # Plot prediction
        # for f in self.config[W_SETTINGS][W_FEATURES]:

        #     # Create var folder
        #     if not os.path.exists(self.pred_dir + "/" + str(f) + "/"):
        #         os.makedirs(self.pred_dir + "/" + str(f) + "/")

        #     f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(f)

        #     for t in tqdm(range(len(yp_npy)), desc = f):
        #         # # test X
        #         # X_t = np.squeeze(X[t,:,:])
        #         # X_t = scaler.inverse_transform(X_t)
        #         # x_npy.append(X_t)

        #         # # test y
        #         # Y_t = np.squeeze(y[t,:,:])
        #         # Y_t = scaler.inverse_transform(Y_t)
        #         # ya_npy.append(Y_t)

        #         # # pred y
        #         # predY_t = np.squeeze(predY[t,:,:])
        #         # predY_t = scaler.inverse_transform(predY_t)
        #         # yp_npy.append(predY_t)

        #         plt.plot(range(t, t + len(x_npy[t][:, f_idx])), x_npy[:, f_idx], color = 'green', label = "past")
        #         plt.plot(range(t - 1 + len(x_npy[t][:, f_idx]), t - 1 + len(x_npy[:, f_idx]) + len(ya_npy[t][:, f_idx])), ya_npy[:, f_idx], color = 'blue', label = "actual")
        #         plt.plot(range(t - 1 + len(x_npy[t][:, f_idx]), t - 1 + len(x_npy[:, f_idx]) + len(yp_npy[t][:, f_idx])), yp_npy[:, f_idx], color = 'red', label = "pred")
        #         plt.title("Multi-step prediction - " + f)
        #         plt.xlabel("step = 0.1s")
        #         plt.ylabel(f)
        #         plt.grid()
        #         plt.legend()
        #         plt.savefig(self.pred_dir + "/" + str(f) + "/" + str(t) + ".png")

        #         plt.clf()
                
        # plt.close()


    def save_cmatrix(self):
        if self.config[W_INPUTATT][W_USECAUSAL]:
            layers = self.model.layers
            ca_matrix = [layers[l].selfatt.causal.numpy() for l in range(1, len(layers))]
            print(ca_matrix)
            print(self.config[W_INPUTATT][W_CMATRIX])

            with open(self.model_dir + '/cmatrix.npy', 'wb') as file_pi:
                np.save(file_pi, ca_matrix)



    def plot_history(self, history):       
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


    def plot_RMSE(self, rmse, show = False):
        plt.figure()
        plt.title("Mean RMSE vs time steps")
        plt.plot(range(self.config[W_SETTINGS][W_NFUTURE]), rmse)
        plt.ylabel("Mean RMSE")
        plt.xlabel("Time steps")
        plt.grid()
        if show:
            plt.show()
        else:
            plt.savefig(self.plot_dir + "/rmse_pred.png", dpi = 300)
            plt.savefig(self.plot_dir + "/rmse_pred.eps", dpi = 300)
        plt.close()


    def mean_RMSE(self, rmse):
        with open(self.model_dir + '/mean_rmse.npy', 'wb') as file:
            np.save(file, np.mean(rmse))


    def plot_prediction(self, x, ya, yp, target_var = None):
        plt.figure()
        if target_var is None:
            for f in self.config[W_SETTINGS][W_FEATURES]:

                # Create var folder
                if not os.path.exists(self.pred_dir + "/" + str(f) + "/"):
                    os.makedirs(self.pred_dir + "/" + str(f) + "/")

                f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(f)

                for t in tqdm(range(len(yp)), desc = f):
                    plt.plot(range(t, t + len(x[t][:, f_idx])), x[:, f_idx], color = 'green', label = "past")
                    plt.plot(range(t - 1 + len(x[t][:, f_idx]), t - 1 + len(x[t][:, f_idx]) + len(ya[t][:, f_idx])), ya[t][:, f_idx], color = 'blue', label = "actual")
                    plt.plot(range(t - 1 + len(x[t][:, f_idx]), t - 1 + len(x[t][:, f_idx]) + len(yp[t][:, f_idx])), yp[t][:, f_idx], color = 'red', label = "pred")
                    plt.title("Multi-step prediction - " + f)
                    plt.xlabel("step = 0.1s")
                    plt.ylabel(f)
                    plt.grid()
                    plt.legend()
                    plt.savefig(self.pred_dir + "/" + str(f) + "/" + str(t) + ".png")

                    plt.clf()
        else:
            # Create var folder
            if not os.path.exists(self.pred_dir + "/" + str(target_var) + "/"):
                os.makedirs(self.pred_dir + "/" + str(target_var) + "/")

            f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(target_var)

            for t in tqdm(range(len(yp)), desc = target_var):
                plt.plot(range(t, t + len(x[t][:, f_idx])), x[t][:, f_idx], color = 'green', label = "past")
                plt.plot(range(t - 1 + len(x[t][:, f_idx]), t - 1 + len(x[t][:, f_idx]) + len(ya[t])), ya[t], color = 'blue', label = "actual")
                plt.plot(range(t - 1 + len(x[t][:, f_idx]), t - 1 + len(x[t][:, f_idx]) + len(yp[t])), yp[t], color = 'red', label = "pred")
                plt.title("Multi-step prediction - " + target_var)
                plt.xlabel("step = 0.1s")
                plt.ylabel(target_var)
                plt.grid()
                plt.legend()
                plt.savefig(self.pred_dir + "/" + str(target_var) + "/" + str(t) + ".png")

                plt.clf()
        plt.close()