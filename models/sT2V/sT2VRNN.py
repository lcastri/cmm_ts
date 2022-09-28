from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model
from models.MyModel import MyModel
from .T2VRNN import T2VRNN
from matplotlib import pyplot as plt
from models.words import *
from tqdm import tqdm
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import os


class sT2VRNN(MyModel):
    def __init__(self, config, target_var, loss, optimizer, metrics):
        super().__init__(config)
        self.target_var = target_var
        self.model = self.create_model(loss, optimizer, metrics)
        plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)


    def create_model(self, loss, optimizer, metrics) -> Model:
        inp = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
        x = T2VRNN(self.config, self.target_var)(inp)
    
        m = Model(inp, x)
        m.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        m.summary()
        return m


    def RMSE(self, X, y, scaler, show = False):
        print('\n##')
        print('## Prediction evaluation through RMSE')
        print('##')

        t_idx = self.config[W_SETTINGS][W_FEATURES].index(self.target_var)
        dummy_y = np.zeros(shape = (y.shape[1], 8))
        
        predY = self.model.predict(X)
        rmse = np.zeros(shape = (1, y.shape[1]))
        for t in tqdm(range(len(y)), desc = 'RMSE'):
            
            # Invert scaling actual
            actualY_t = np.squeeze(y[t,:,:])
            dummy_y[:, t_idx] = actualY_t 
            actualY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
            actualY_t = np.reshape(actualY_t, (actualY_t.shape[0], 1))

            # Invert scaling pred
            predY_t = np.squeeze(predY[t,:,:])
            dummy_y[:, t_idx] = predY_t
            predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
            predY_t = np.reshape(predY_t, (predY_t.shape[0], 1))

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

        with open(self.model_dir + '/rmse', 'wb') as file_pi:
            np.save(file_pi, rmse_mean)
        return rmse_mean
        

    def plot_predictions(self, X, y, scaler):
        print('\n##')
        print('## Predictions')
        print('##')

        t_idx = self.config[W_SETTINGS][W_FEATURES].index(self.target_var)
        dummy_y = np.zeros(shape = (y.shape[1], 8))

        predY = self.model.predict(X)

        # Create var folder
        if not os.path.exists(self.pred_dir + "/" + str(self.target_var) + "/"):
            os.makedirs(self.pred_dir + "/" + str(self.target_var) + "/")

        f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(self.target_var)

        for t in tqdm(range(len(predY)), desc = self.target_var):
            # test X
            X_t = np.squeeze(X[t,:,:])
            X_t = scaler.inverse_transform(X_t)

            # test y
            Y_t = np.squeeze(y[t,:,:])
            dummy_y[:, t_idx] = Y_t 
            Y_t = scaler.inverse_transform(dummy_y)[:, t_idx]

            # pred y
            predY_t = np.squeeze(predY[t,:,:])
            dummy_y[:, t_idx] = predY_t
            predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]

            plt.plot(range(t, t + len(X_t[:, f_idx])), X_t[:, f_idx], color = 'green', label = "past")
            plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(Y_t)), Y_t, color = 'blue', label = "actual")
            plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(predY_t)), predY_t, color = 'red', label = "pred")
            plt.title("Multi-step prediction - " + self.target_var)
            plt.xlabel("step = 0.1s")
            plt.ylabel(self.target_var)
            plt.grid()
            plt.legend()
            plt.savefig(self.pred_dir + "/" + str(self.target_var) + "/" + str(t) + ".png")

            plt.clf()
                
        plt.close()