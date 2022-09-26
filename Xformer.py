# Machine learning
from keras.layers import *
from keras.models import *
from kerashypetune import KerasGridSearch
from keras import backend as K
from parameters import *
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
    # x = RepeatVector(N_FUTURE)(x)
    # x = LSTM(param['LSTMunit'], return_sequences = True, activation = param['LSTMact'])(x)
    # x = TimeDistributed(Dense(param['Dunit'], activation = param['Dact']))(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dense(N_FUTURE, activation = 'linear')(x)
    
    m = Model(inp, x)
    # m.compile(loss='mse', optimizer = 'adam', metrics=['mse', 'mae', 'mape', 'accuracy'])
    m.compile(loss='mse', optimizer = 'adam')
    
    return m


# Data initialization
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.downsample(10)
X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries('d_g')
create_folder(MODEL_FOLDER + "/")


param_grid = {
    'LSTMunit': [128, 64, 32],
    # 'Dunit': [128, 64, 32],
    't2v_dim': [128, 64, 16],
    'lr': [1e-3, 1e-4, 1e-5], 
    'LSTMact': ['tanh'], 
    # 'Dact': ['linear', 'relu'], 
    'epochs': 50,
    'batch_size': [128, 256]
}

hypermodel = lambda x: T2V_NN(param = x)

# kgs_t2v = KerasGridSearch(hypermodel, param_grid, monitor='val_loss', greater_is_better = False, tuner_verbose = 1)
# kgs_t2v.search(X_train, y_train, validation_data = (X_val, y_val), shuffle = False)
best_param = {
    'LSTMunit': 128,
    'Dunit': 32,
    't2v_dim': 64,
    'lr': 0.00001, 
    'LSTMact': 'tanh', 
    'Dact': 'relu', 
    'epochs': 200,
    'batch_size': 128
}
model = T2V_NN(param=best_param)
model.summary()
cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_FOLDER + '/', save_best_only = True)
history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = best_param['epochs'],
                    shuffle = False, callbacks = [cb_checkpoints, cb_earlystop])
plt.figure()
plt.plot(history.history["loss"], label = "Training loss")
plt.plot(history.history["val_loss"], label = "Validation loss")
plt.legend()
plt.savefig(MODEL_FOLDER + "/plots/loss.png", dpi = 300)
plt.savefig(MODEL_FOLDER + "/plots/loss.eps", dpi = 300)

plt.figure()
plt.plot(history.history["mae"], label = "Training mae")
plt.plot(history.history["val_mae"], label = "Validation mae")
plt.legend()
plt.savefig(MODEL_FOLDER + "/plots/mae.png", dpi = 300)
plt.savefig(MODEL_FOLDER + "/plots/mae.eps", dpi = 300)

plt.figure()
plt.plot(history.history["mape"], label = "Training mape")
plt.plot(history.history["val_mape"], label = "Validation mape")
plt.legend()
plt.savefig(MODEL_FOLDER + "/plots/mape.png", dpi = 300)
plt.savefig(MODEL_FOLDER + "/plots/mape.eps", dpi = 300)

plt.figure()
plt.plot(history.history["accuracy"], label = "Training accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
plt.legend()
plt.savefig(MODEL_FOLDER + "/plots/accuracy.png", dpi = 300)
plt.savefig(MODEL_FOLDER + "/plots/accuracy.eps", dpi = 300)

with open(MODEL_FOLDER + '/history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
