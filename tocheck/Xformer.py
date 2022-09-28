# Machine learning
import tarfile
from keras.layers import *
from keras.models import *
from kerashypetune import KerasGridSearch
from keras import backend as K
from constants import *
import os
from Data import Data
import pickle
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) 

from models import utils
from tqdm import tqdm
from math import sqrt
from sklearn.metrics import mean_squared_error

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder + "plots"):
        os.makedirs(folder + "plots")

class T2V(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.P = self.add_weight(name='P',
                                shape=(input_shape[1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.w = self.add_weight(name='w',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        self.p = self.add_weight(name='p',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        super(T2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p #if i = 0
        sin_trans = K.sin(K.dot(x, self.W) + self.P) # Frequecy and phase shift of sine function, learnable parameters. if 1 <= i <= k
        
        return K.concatenate([sin_trans, original], -1)


def T2V_NN(param):
    inp = Input(shape = (N_PAST, N_FEATURES))
    x = T2V(param['t2v_dim'])(inp)
    x = LSTM(param['LSTMunit'], activation = param['LSTMact'])(x)
    x = Dense(param['Dunit'], activation = param['Dact'])(x)
    x = Dense(N_FUTURE, activation = 'linear')(x)
    
    m = Model(inp, x)
    m.compile(loss='mse', optimizer = 'adam')
    
    return m


def RMSE(X, y, scaler, show = False):
    print('\n##')
    print('## Prediction evaluation through RMSE')
    print('##')

    plot_dir = utils.create_plot_dir(MODEL_FOLDER)

    t_idx = list(d.features).index('d_g')
    dummy_y = np.zeros(shape = (y.shape[1], 8))
        
    predY = model.predict(X)
    rmse = np.zeros(shape = (1, y.shape[1]))
    for t in tqdm(range(len(y)), desc = 'RMSE'):
            
        # Invert scaling actual
        actualY_t = np.squeeze(y[t,:,:])
        dummy_y[:, t_idx] = actualY_t 
        actualY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
        actualY_t = np.reshape(actualY_t, (actualY_t.shape[0], 1))

        # Invert scaling pred
        predY_t = np.squeeze(predY[t,:])
        dummy_y[:, t_idx] = predY_t
        predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]
        predY_t = np.reshape(predY_t, (predY_t.shape[0], 1))

        rmse = rmse + np.array([sqrt(mean_squared_error(actualY_t[f], predY_t[f])) for f in range(N_FUTURE)])
    rmse_mean = np.sum(rmse, axis=0)/len(y)

    plt.figure()
    plt.title("Mean RMSE vs time steps")
    plt.plot(range(N_FUTURE), rmse_mean)
    plt.xlabel("Time steps")
    plt.xlabel("Mean RMSE")
    if show:
        plt.show()
    else:
        plt.savefig(plot_dir + "/rmse_pred.png", dpi = 300)
        plt.savefig(plot_dir + "/rmse_pred.eps", dpi = 300)
    plt.close()
    return rmse_mean
        

def plot_predictions(X, y, scaler):
    print('\n##')
    print('## Predictions')
    print('##')

    f = 'd_g'
    t_idx = list(d.features).index('d_g')
    dummy_y = np.zeros(shape = (y.shape[1], 8))

    # Create prediction folder
    dir_pred = utils.create_pred_dir(MODEL_FOLDER)

    predY = model.predict(X)

        # Create var folder
    if not os.path.exists(dir_pred + "/" + str(f) + "/"):
        os.makedirs(dir_pred + "/" + str(f) + "/")


    for t in tqdm(range(len(predY)), desc = f):
        # test X
        X_t = np.squeeze(X[t,:,:])
        X_t = scaler.inverse_transform(X_t)

        # test y
        Y_t = np.squeeze(y[t,:,:])
        dummy_y[:, t_idx] = Y_t 
        Y_t = scaler.inverse_transform(dummy_y)[:, t_idx]

        # pred y
        predY_t = np.squeeze(predY[t,:])
        dummy_y[:, t_idx] = predY_t
        predY_t = scaler.inverse_transform(dummy_y)[:, t_idx]

        plt.plot(range(t, t + len(X_t[:, t_idx])), X_t[:, t_idx], color = 'green', label = "past")
        plt.plot(range(t - 1 + len(X_t[:, t_idx]), t - 1 + len(X_t[:, t_idx]) + len(Y_t)), Y_t, color = 'blue', label = "actual")
        plt.plot(range(t - 1 + len(X_t[:, t_idx]), t - 1 + len(X_t[:, t_idx]) + len(predY_t)), predY_t, color = 'red', label = "pred")
        plt.title("Multi-step prediction - " + f)
        plt.xlabel("step = 0.1s")
        plt.ylabel(f)
        plt.legend()
        plt.savefig(dir_pred + "/" + str(f) + "/" + str(t) + ".png")

        plt.clf()
                
    plt.close()


# Data initialization
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = 'd_g')
d.downsample(10)
X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()
create_folder(MODEL_FOLDER + "/")


# param_grid = {
#     'LSTMunit': [128, 64, 32],
#     'Dunit': [256, 128, 64, 32],
#     't2v_dim': [128, 64, 16],
#     'lr': [1e-3, 1e-4, 1e-5], 
#     'LSTMact': ['tanh'], 
#     'Dact': ['linear', 'relu'], 
#     'epochs': 50,
#     'batch_size': [128, 256, 512]
# }

# hypermodel = lambda x: T2V_NN(param = x)

# kgs_t2v = KerasGridSearch(hypermodel, param_grid, monitor='val_loss', greater_is_better = False, tuner_verbose = 1)
# best_param = kgs_t2v.search(X_train, y_train, validation_data = (X_val, y_val), shuffle = False)['best_params']
best_param = {
    'LSTMunit': 128,
    'Dunit': 32,
    't2v_dim': 64,
    'lr': 0.00001, 
    'LSTMact': 'tanh', 
    'Dact': 'relu', 
    'epochs': 2,
    'batch_size': 128
}
model = T2V_NN(param = best_param)
model.summary()
cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_FOLDER + '/', save_best_only = True)
history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = best_param['epochs'],
                    shuffle = False, callbacks = [cb_checkpoints, cb_earlystop], batch_size = best_param['batch_size'])
plt.figure()
plt.plot(history.history["loss"], label = "Training loss")
plt.plot(history.history["val_loss"], label = "Validation loss")
plt.legend()
plt.savefig(MODEL_FOLDER + "/plots/loss.png", dpi = 300)
plt.savefig(MODEL_FOLDER + "/plots/loss.eps", dpi = 300)
plt.close()

with open(MODEL_FOLDER + '/history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

RMSE(x_test, y_test, d.scaler)
plot_predictions(x_test, y_test, d.scaler)