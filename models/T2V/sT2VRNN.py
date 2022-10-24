from keras.layers import *
from keras.models import *
from models.MyModel import MyModel
from models.utils import Models
from .T2VRNN import T2VRNN
import models.Words as W


class sT2VRNN(MyModel):
    def __init__(self, config : dict = None, folder : str = None):
        super().__init__(name = Models.sT2V, config = config, folder = folder)


    def create_model(self, target_var, loss, optimizer, metrics, searchBest = False) -> Model:
        self.target_var = target_var

        inp = Input(shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
        x = T2VRNN(self.config, target_var, name = target_var + "_T2V", searchBest = searchBest)(inp)
    
        self.model = Model(inp, x)
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        self.model.summary()
        # plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)
        return self.model


    # def MAE(self, X, y, scaler, show = False):
    #     print('\n##')
    #     print('## Prediction evaluation through MAE')
    #     print('##')

    #     t_idx = self.config[W.FEATURES].index(self.target_var)
    #     dummy_y = np.zeros(shape = (y.shape[1], 8))
        
    #     predY = self.model.predict(X)
    #     mae = np.zeros(shape = (y.shape[1], 1))
    #     for t in tqdm(range(len(y)), desc = 'Abs error'):
            
    #         # Invert scaling actual
    #         actualY_t = np.squeeze(y[t,:,:])
    #         dummy_y[:, t_idx] = actualY_t 
    #         actualY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
    #         actualY_t = np.reshape(actualY_t, (actualY_t.shape[0], 1))

    #         # Invert scaling pred
    #         predY_t = np.squeeze(predY[t,:,:])
    #         dummy_y[:, t_idx] = predY_t
    #         predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
    #         predY_t = np.reshape(predY_t, (predY_t.shape[0], 1))

    #         mae = mae + abs(actualY_t - predY_t)
    #     mae_mean = mae/len(y)

    #     with open(self.model_dir + '/mae.npy', 'wb') as file:
    #         np.save(file, mae_mean)

    #     self.plot_MAE(mae_mean, show = show)
    #     return mae_mean
        

    # def predict(self, X, y, scaler, plot = False):
    #     print('\n##')
    #     print('## Predictions')
    #     print('##')
    #     x_npy = list()
    #     ya_npy = list()
    #     yp_npy = list()

    #     t_idx = self.config[W.FEATURES].index(self.target_var)
    #     dummy_y = np.zeros(shape = (y.shape[1], 8))

    #     predY = self.model.predict(X)

    #     for t in range(len(predY)):
    #         # test X
    #         X_t = np.squeeze(X[t,:,:])
    #         X_t = scaler.inverse_transform(X_t)
    #         x_npy.append(X_t)

    #         # test y
    #         Y_t = np.squeeze(y[t,:,:])
    #         dummy_y[:, t_idx] = Y_t 
    #         Y_t = scaler.inverse_transform(dummy_y)[:, t_idx]
    #         ya_npy.append(Y_t)

    #         # pred y
    #         predY_t = np.squeeze(predY[t,:,:])
    #         dummy_y[:, t_idx] = predY_t
    #         predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
    #         yp_npy.append(predY_t)
        
    #     with open(self.pred_dir + '/x_npy.npy', 'wb') as file:
    #         np.save(file, x_npy)
    #     with open(self.pred_dir + '/ya_npy.npy', 'wb') as file:
    #         np.save(file, ya_npy)
    #     with open(self.pred_dir + '/yp_npy.npy', 'wb') as file:
    #         np.save(file, yp_npy)


    #     if plot: self.plot_prediction(x_npy, ya_npy, yp_npy, target_var = self.target_var)
