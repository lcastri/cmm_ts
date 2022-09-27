from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model
from models.MyModel import MyModel
from .CNNIAED import CNNIAED
from models.words import *


class mIACNNED(MyModel):
    def __init__(self, config, target_var):
        super().__init__(config)
        self.target_var = target_var
        self.model = self.create_model()
        plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)


    def create_model(self) -> Model:
        inp = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
        
        # Multihead
        channels = list()
        for var in self.config[W_SETTINGS][W_FEATURES]:
            channels.append(CNNIAED(self.config, var, name = var + "_CNNIAED")(inp))

        # Concatenation
        y = concatenate(channels, axis = 2)
    
        m = Model(inp, y)
        m.compile(loss='mse', optimizer = 'adam', metrics=['mse', 'mae', 'mape', 'accuracy'])
        # m.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse'])
        # m.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'])

        m.summary()
        return m


# import os
# import pickle
# from keras.layers import *
# from keras.models import *
# from .CNNIAED import CNNIAED
# from matplotlib import pyplot as plt
# from models.words import *
# import models.utils as utils
# from tqdm import tqdm
# import numpy as np
# from math import sqrt
# from sklearn.metrics import mean_squared_error
# from parameters import ROOT_DIR


# class mIACNNED(Model):
#     def __init__(self, config):
#         super(mIACNNED, self).__init__()
#         self.config = config
#         self.channels = dict()

#         self.dir_plot = utils.create_plot_dir(self.config[W_SETTINGS][W_FOLDER])
#         utils.no_warning()

#         # Multihead
#         for var in self.config[W_SETTINGS][W_FEATURES]:
#             self.channels[var] = CNNIAED(self.config, var, name = var + "_IAED")

#         # Concatenation
#         self.concat = Concatenate(axis = 1)

#         # Reshape
#         self.reshape = Reshape((-1, self.config[W_SETTINGS][W_NFEATURES]))

#         # LSTM
#         self.lstm_1 = LSTM(self.config[W_ENC][0][W_UNITS], 
#                            activation = self.config[W_ENC][0][W_ACT],
#                            return_sequences = self.config[W_ENC][0][W_RSEQ],
#                            return_state = self.config[W_ENC][0][W_RSTATE])

#         # Repeat
#         self.repeat = RepeatVector(self.config[W_SETTINGS][W_NFUTURE])

#         # LSTM
#         self.lstm_2 = LSTM(self.config[W_DEC][0][W_UNITS], 
#                            activation = self.config[W_DEC][0][W_ACT],
#                            return_sequences = self.config[W_DEC][0][W_RSEQ],
#                            return_state = self.config[W_DEC][0][W_RSTATE])


#         # Dense
#         self.outdense = list()
#         for i in range(len(self.config[W_OUT])):
#             self.outdense.append(Dense(self.config[W_OUT][i][W_UNITS], activation = self.config[W_OUT][i][W_ACT], 
#                                        name = self.target_var + '_D'))
#         self.out = Dense(self.config[W_SETTINGS][W_NFUTURE], activation = 'linear', name = self.target_var + '_out')

    
#     def call(self, x):
#         y = self.concat([self.channels[var](x) for var in self.config[W_SETTINGS][W_FEATURES]])
#         y = self.reshape(y)
#         y = self.lstm_1(y)
#         y = self.repeat(y)
#         y = self.lstm_2(y)
#         for i in range(len(self.config[W_OUT])): y = self.outdense[i](y)
#         y = self.out(y)

#         return y

    
#     def model(self):
#         x = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
#         return Model(inputs = [x], outputs = self.call(x))


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

#         if "mae" in history.history.keys():
#             plt.figure()
#             plt.plot(history.history["mae"], label = "Training mae")
#             plt.plot(history.history["val_mae"], label = "Validation mae")
#             plt.legend()
#             plt.savefig(self.dir_plot + "/mae.png", dpi = 300)
#             plt.savefig(self.dir_plot + "/mae.eps", dpi = 300)

#         if "mape" in history.history.keys():
#             plt.figure()
#             plt.plot(history.history["mape"], label = "Training mape")
#             plt.plot(history.history["val_mape"], label = "Validation mape")
#             plt.legend()
#             plt.savefig(self.dir_plot + "/mape.png", dpi = 300)
#             plt.savefig(self.dir_plot + "/mape.eps", dpi = 300)

#         if "accuracy" in history.history.keys():
#             plt.figure()
#             plt.plot(history.history["accuracy"], label = "Training accuracy")
#             plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
#             plt.legend()
#             plt.savefig(self.dir_plot + "/accuracy.png", dpi = 300)
#             plt.savefig(self.dir_plot + "/accuracy.eps", dpi = 300)

#         with open(ROOT_DIR + '/' + self.config[W_SETTINGS][W_FOLDER] + '/history', 'wb') as file_pi:
#             pickle.dump(history.history, file_pi)


#     def RMSE(self, X, y, scaler, show = False):
#         print('\n##')
#         print('## Prediction evaluation through RMSE')
#         print('##')

#         predY = self.predict(X)
#         rmse = np.zeros(shape = (1, y.shape[1]))
#         for t in tqdm(range(len(y)), desc = 'RMSE'):
#             actualY_t = np.squeeze(y[t,:,:])
#             predY_t = np.squeeze(predY[t,:,:])
#             actualY_t = scaler.inverse_transform(actualY_t)
#             predY_t = scaler.inverse_transform(predY_t)
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

#         # Create prediction folder
#         dir_pred = utils.create_pred_dir(self.config[W_SETTINGS][W_FOLDER])

#         predY = self.predict(X)
#         for f in self.config[W_SETTINGS][W_FEATURES]:

#             # Create var folder
#             if not os.path.exists(dir_pred + "/" + str(f) + "/"):
#                 os.makedirs(dir_pred + "/" + str(f) + "/")

#             f_idx = list(self.config[W_SETTINGS][W_FEATURES]).index(f)

#             for t in tqdm(range(len(predY)), desc = f):
#                 # test X
#                 X_t = np.squeeze(X[t,:,:])
#                 X_t = scaler.inverse_transform(X_t)

#                 # test y
#                 Y_t = np.squeeze(y[t,:,:])
#                 Y_t = scaler.inverse_transform(Y_t)

#                 # pred y
#                 predY_t = np.squeeze(predY[t,:,:])
#                 predY_t = scaler.inverse_transform(predY_t)

#                 plt.plot(range(t, t + len(X_t[:, f_idx])), X_t[:, f_idx], color = 'green', label = "past")
#                 plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(Y_t[:, f_idx])), Y_t[:, f_idx], color = 'blue', label = "actual")
#                 plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(predY_t[:, f_idx])), predY_t[:, f_idx], color = 'red', label = "pred")
#                 plt.title("Multi-step prediction - " + f)
#                 plt.xlabel("step = 0.1s")
#                 plt.ylabel(f)
#                 plt.legend()
#                 plt.savefig(dir_pred + "/" + str(f) + "/" + str(t) + ".png")

#                 plt.clf()
                
#         plt.close()