from models.utils import *
from models.AdjLR import AdjLR
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from Data import Data
from keras.layers import *
from keras.models import *
from constants import *

# Models import
from models.mIAED import mIAED
from models.sIAED import sIAED
from models.sT2VRNN import sT2VRNN
from models.configIAED import config as cIAED
from models.configT2V import config as cT2V
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

df, features = get_df(11)

# Parameters definition
MODEL = Models.sT2V.value
TARGETVAR = 'd_g' if MODEL == Models.sIAED.value or MODEL == Models.sT2V.value else None 
N_FUTURE = 48
N_PAST = 32
N_DELAY = 0
TRAIN_PERC = 0.7
VAL_PERC = 0.1
TEST_PERC = 0.2
MODEL_FOLDER = "PROVA9"
BATCH_SIZE = 32
PATIENCE = 25
EPOCH = 500

if MODEL == Models.sIAED.value:
    if TARGETVAR == None: raise ValueError('for models sIAED, target_var needs to be specified')
    # Single-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    # d.plot_ts()
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # IAED Model definition
    config = init_config(cIAED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = True,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = sIAED(config = config)
    model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


elif MODEL == Models.sT2V.value:
    if TARGETVAR == None: raise ValueError('for models sT2V, target_var needs to be specified')
    # Single-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # IAED Model definition
    config = init_config(cT2V, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = False,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = sT2VRNN(config = config)
    model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


elif MODEL == Models.mIAED.value:
    # Multi-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
    # d.augment()
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    d.plot_ts()
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # IAED Model definition
    config = init_config(cIAED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = True,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = mIAED(config = config)
    model.create_model(loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


# Model fit
cbs = list()
cbs.append(EarlyStopping(patience = PATIENCE))
cbs.append(ModelCheckpoint('training_result/' + MODEL_FOLDER + '/', save_best_only = True))
model.fit(X = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = BATCH_SIZE, 
          epochs = EPOCH, callbacks = cbs)

model.save_cmatrix()

# Model evaluation
model.MAE(X_test, y_test, d.scaler)

# Model predictions
model.predict(X_test, y_test, d.scaler, plot = True)
