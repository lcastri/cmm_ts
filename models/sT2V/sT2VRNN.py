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
    def __init__(self, config, target_var):
        super().__init__(config)
        self.target_var = target_var
        self.model = self.create_model()
        plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)


    def create_model(self) -> Model:
        inp = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
        x = T2VRNN(self.config, self.target_var)(inp)
    
        m = Model(inp, x)
        m.compile(loss='mse', optimizer = 'adam', metrics=['mse', 'mae', 'mape', 'accuracy'])
        # m.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse'])
        # m.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'])

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


# class sT2VRNN(Model):
#     def __init__(self, config, target_var):
#         super(sT2VRNN, self).__init__()
#         self.config = config
#         self.target_var = target_var
#         self.channels = dict()

#         self.dir_plot = utils.create_plot_dir(self.config[W_SETTINGS][W_FOLDER])
#         utils.no_warning()

#         # Model definition
#         self.inp = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
#         self.channels = T2VRNN(self.config, target_var, name = target_var + "_T2VRNN")
    

#     def call(self, input_sequence):
#         x = input_sequence
#         return self.channels(x)

    
#     def model(self):
#         return Model(inputs = [self.inp], outputs = self.call(self.inp))


#     def get_config(self):
#         data = dict()
#         if self.config[W_SETTINGS][W_USEATT] and self.config[W_INPUTATT][W_USECAUSAL]:
#             for var in self.config[W_SETTINGS][W_FEATURES]:
#                 data[var] = self.channels[var].ca.causal.numpy()
#         return {"causal_weights": data}


#     def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
#         history = super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
        
#         if "loss" in history.history.keys():
#             plt.figure()
#             plt.plot(history.history["loss"], label = "Training loss")
#             plt.plot(history.history["val_loss"], label = "Validation loss")
#             plt.legend()
#             plt.savefig(self.dir_plot + "/loss.png", dpi = 300)
#             plt.savefig(self.dir_plot + "/loss.eps", dpi = 300)
#             plt.close()

#         if "mae" in history.history.keys():
#             plt.figure()
#             plt.plot(history.history["mae"], label = "Training mae")
#             plt.plot(history.history["val_mae"], label = "Validation mae")
#             plt.legend()
#             plt.savefig(self.dir_plot + "/mae.png", dpi = 300)
#             plt.savefig(self.dir_plot + "/mae.eps", dpi = 300)
#             plt.close()

#         if "mape" in history.history.keys():
#             plt.figure()
#             plt.plot(history.history["mape"], label = "Training mape")
#             plt.plot(history.history["val_mape"], label = "Validation mape")
#             plt.legend()
#             plt.savefig(self.dir_plot + "/mape.png", dpi = 300)
#             plt.savefig(self.dir_plot + "/mape.eps", dpi = 300)
#             plt.close()

#         if "accuracy" in history.history.keys():
#             plt.figure()
#             plt.plot(history.history["accuracy"], label = "Training accuracy")
#             plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
#             plt.legend()
#             plt.savefig(self.dir_plot + "/accuracy.png", dpi = 300)
#             plt.savefig(self.dir_plot + "/accuracy.eps", dpi = 300)
#             plt.close()

#         with open(ROOT_DIR + '/' + self.config[W_SETTINGS][W_FOLDER] + '/history', 'wb') as file_pi:
#             pickle.dump(history.history, file_pi)


#     def RMSE(self, X, y, scaler, show = False):
#         print('\n##')
#         print('## Prediction evaluation through RMSE')
#         print('##')

#         t_idx = self.config[W_SETTINGS][W_FEATURES].index(self.target_var)
#         dummy_y = np.zeros(shape = (y.shape[1], 8))
        
#         predY = self.predict(X)
#         rmse = np.zeros(shape = (1, y.shape[1]))
#         for t in tqdm(range(len(y)), desc = 'RMSE'):
            
#             # Invert scaling actual
#             actualY_t = np.squeeze(y[t,:,:])
#             dummy_y[:, t_idx] = actualY_t 
#             actualY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
#             actualY_t = np.reshape(actualY_t, (actualY_t.shape[0], 1))

#             # Invert scaling pred
#             predY_t = np.squeeze(predY[t,:,:])
#             dummy_y[:, t_idx] = predY_t
#             predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
#             predY_t = np.reshape(predY_t, (predY_t.shape[0], 1))

#             rmse = rmse + np.array([sqrt(mean_squared_error(actualY_t[f], predY_t[f])) for f in range(self.config[W_SETTINGS][W_NFUTURE])])
#         rmse_mean = np.sum(rmse, axis=0)/len(y)

#         plt.figure()
#         plt.title("Mean RMSE vs time steps")
#         plt.plot(range(self.config[W_SETTINGS][W_NFUTURE]), rmse_mean)
#         plt.xlabel("Time steps")
#         plt.xlabel("Mean RMSE")
#         if show:
#             plt.show()
#         else:
#             plt.savefig(self.dir_plot + "/rmse_pred.png", dpi = 300)
#             plt.savefig(self.dir_plot + "/rmse_pred.eps", dpi = 300)
#         plt.close()
#         return rmse_mean
        

#     def plot_predictions(self, X, y, scaler):
#         print('\n##')
#         print('## Predictions')
#         print('##')

#         t_idx = self.config[W_SETTINGS][W_FEATURES].index(self.target_var)
#         dummy_y = np.zeros(shape = (y.shape[1], 8))

#         # Create prediction folder
#         dir_pred = utils.create_pred_dir(self.config[W_SETTINGS][W_FOLDER])

#         predY = self.predict(X)

#         # Create var folder
#         if not os.path.exists(dir_pred + "/" + str(self.target_var) + "/"):
#             os.makedirs(dir_pred + "/" + str(self.target_var) + "/")

#         f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(self.target_var)

#         for t in tqdm(range(len(predY)), desc = self.target_var):
#             # test X
#             X_t = np.squeeze(X[t,:,:])
#             X_t = scaler.inverse_transform(X_t)

#             # test y
#             Y_t = np.squeeze(y[t,:,:])
#             dummy_y[:, t_idx] = Y_t 
#             Y_t = scaler.inverse_transform(dummy_y)[:, t_idx]

#             # pred y
#             predY_t = np.squeeze(predY[t,:,:])
#             dummy_y[:, t_idx] = predY_t
#             predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]

#             plt.plot(range(t, t + len(X_t[:, f_idx])), X_t[:, f_idx], color = 'green', label = "past")
#             plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(Y_t)), Y_t, color = 'blue', label = "actual")
#             plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(predY_t)), predY_t, color = 'red', label = "pred")
#             plt.title("Multi-step prediction - " + self.target_var)
#             plt.xlabel("step = 0.1s")
#             plt.ylabel(self.target_var)
#             plt.legend()
#             plt.savefig(dir_pred + "/" + str(self.target_var) + "/" + str(t) + ".png")

#             plt.clf()
                
#         plt.close()