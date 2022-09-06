from model import CA_LSTM
from parameters import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Data import Data
import numpy as np


CM_PCMCI = np.array([[0.632,0.065,0.125,0.088,0.138,0.108,0.06,0.048], 
               [0.078,0.274,0.092,0.094,0.103,0.08,0.068,0.049], 
               [0.09,0.196,0.27,0.106,0.137,0.111,0.065,0.049], 
               [0.1,0.072,0.122,0.166,0.154,0.102,0.059,0], 
               [0.111,0.059,0.095,0.087,0.131,0.132,0.054,0.057], 
               [0.122,0.067,0.083,0.112,0.418,0.541,0,0], 
               [0.086,0,0.063,0.062,0.13,0.076,0,0.052], 
               [0,0,0.074,0.067,0.05,0,0.051,0.708]])

CM_FPCMCI = np.array([[0.794693885975173,0.0797596212634794,0,0,0.207147494884196,0,0,0],
               [0,0.547118275252963,0.118972264307896,0,0,0,0,0],
               [0,0.411013790019703,0.398058466007042,0,0,0,0,0.0622522550249479],
               [0,0,0,0.400374937278639,0.147993676497357,0,0.0963358126783955,0.120359691147529],
               [0,0,0,0,0.116815916574682,0,0,0],
               [0,0,0,0,0.239283926475173,0.60894990342525,0,0],
               [0,0,0,0,0,0,0.991103983176182,0],
               [0,0.0684969047836938,0.0634190046412317,0,0,0,0,0.972004088748988]])

cb_earlystop = EarlyStopping(monitor='loss', patience = 3)
cb_checkpoints = ModelCheckpoint(MODEL_NAME + '/', save_best_only=True)

net = CA_LSTM(list(df.columns), 64, N_PAST, use_attention_layer = True, causal_matrix = CM_FPCMCI)
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.scale_data()
X_train, y_train, X_val, y_val, _, _ = d.get_timeseries()

net.fit(50, (X_train, y_train), (X_val, y_val), callbacks = [cb_checkpoints, cb_earlystop])
