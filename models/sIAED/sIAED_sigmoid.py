from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model
from models.MyModel import MyModel
from .IAEDsimple_sigmoid import IAED
from models.words import *
from tqdm import tqdm
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


class sIAED(MyModel):
    def __init__(self, config : dict = None, folder : str = None):
        super().__init__(config = config, folder = folder)
               

    def create_model(self, target_var, loss, optimizer, metrics):
        self.target_var = target_var

        inp = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
        x = IAED(self.config, target_var, name = target_var + "_IAED")(inp)
    
        m = Model(inp, x)
        # m.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        m.compile(loss = loss, optimizer = optimizer, metrics = metrics, run_eagerly = True)

        m.summary()
        self.model = m
        plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)


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

        with open(self.model_dir + '/rmse.npy', 'wb') as file:
            np.save(file, rmse_mean)

        self.plot_RMSE(rmse_mean, show = show)
        return rmse_mean
        

    def predict(self, X, y, scaler, plot = False):
        print('\n##')
        print('## Predictions')
        print('##')
        x_npy = list()
        ya_npy = list()
        yp_npy = list()

        t_idx = self.config[W_SETTINGS][W_FEATURES].index(self.target_var)
        dummy_y = np.zeros(shape = (y.shape[1], 8))

        predY = self.model.predict(X)

        for t in range(len(predY)):
            # test X
            X_t = np.squeeze(X[t,:,:])
            X_t = scaler.inverse_transform(X_t)
            x_npy.append(X_t)

            # test y
            Y_t = np.squeeze(y[t,:,:])
            dummy_y[:, t_idx] = Y_t 
            Y_t = scaler.inverse_transform(dummy_y)[:, t_idx]
            ya_npy.append(Y_t)

            # pred y
            predY_t = np.squeeze(predY[t,:,:])
            dummy_y[:, t_idx] = predY_t
            predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
            yp_npy.append(predY_t)
        
        with open(self.pred_dir + '/x_npy.npy', 'wb') as file:
            np.save(file, x_npy)
        with open(self.pred_dir + '/ya_npy.npy', 'wb') as file:
            np.save(file, ya_npy)
        with open(self.pred_dir + '/yp_npy.npy', 'wb') as file:
            np.save(file, yp_npy)


        if plot: self.plot_prediction(x_npy, ya_npy, yp_npy, self.target_var)
