# from model_multistep import CA_LSTM
from notworking.model_multistep2 import CA_LSTM
from IAEncDec import IAEncDec
# from model import CA_LSTM
from parameters import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Data import Data


cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_NAME + '/', save_best_only = True)

net = CA_LSTM(folder = MODEL_NAME + '/',
              features = list(df.columns), 
              list_units = [100, 100], 
              n_past = N_PAST, 
              n_future = N_FUTURE, 
              batch_size = 128, 
              use_attention_layer = False, 
              causal_matrix = CM_FPCMCI,
              causal_config = {'trainable': True, 
                               'constraint_threshold': 0.3})

d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.downsample(10)
d.scale_data()
X_train, y_train, X_val, y_val, _, _ = d.get_timeseries()

net.fit(100, (X_train, y_train), (X_val, y_val), callbacks = [cb_checkpoints, cb_earlystop])
