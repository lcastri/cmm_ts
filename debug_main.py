from models.utils import *
from models.AdjLR import AdjLR
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from Data import Data
from keras.layers import *
from keras.models import *
from constants import *

# IAED import
from models.IAED.mIAED import mIAED
from models.IAED.sIAED import sIAED
from models.IAED.config import config as cIAED
# T2V import
from models.T2V.mT2VRNN import mT2VRNN
from models.T2V.sT2VRNN import sT2VRNN
from models.T2V.config import CONFIG as cT2V
# CNN import
from models.CNNLSTM.mCNNLSTM import mCNNLSTM
from models.CNNLSTM.sCNNLSTM import sCNNLSTM
from models.CNNLSTM.config import CONFIG as cCNN
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

df, features = get_df(11)

# Parameters definition
MODEL = Models.mIAED
TARGETVAR = 'd_g' if MODEL == Models.sIAED or MODEL == Models.sT2V or MODEL == Models.sCNN else None 
N_FUTURE = 48
N_PAST = 32
N_DELAY = 0
TRAIN_PERC = 0.7
VAL_PERC = 0.1
TEST_PERC = 0.2
MODEL_FOLDER = "PROVA12"
BATCH_SIZE = 32
PATIENCE = 25
EPOCH = 1


if MODEL == Models.sIAED:
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
    model = sIAED(df = df, config = config)
    model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


elif MODEL == Models.sT2V:
    if TARGETVAR == None: raise ValueError('for models sT2V, target_var needs to be specified')
    # Single-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # T2V Model definition
    config = init_config(cT2V, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = False,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = sT2VRNN(df = df, config = config)
    model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


elif MODEL == Models.sCNN:
    if TARGETVAR == None: raise ValueError('for models sCNN, target_var needs to be specified')
    # Single-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # CNN Model definition
    config = init_config(cCNN, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = False,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = sCNNLSTM(df = df, config = config)
    model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


elif MODEL == Models.mIAED:
    # Multi-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
    # d.augment()
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    # d.plot_ts()
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # IAED Model definition
    config = init_config(cIAED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = True,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = mIAED(df = df, config = config)
    model.create_model(loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


elif MODEL == Models.mT2V:
    # Multi-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
    # d.augment()
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    # d.plot_ts()
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # IAED Model definition
    config = init_config(cT2V, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = True,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = mT2VRNN(df = df, config = config)
    model.create_model(loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


elif MODEL == Models.mCNN:
    # Multi-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
    # d.augment()
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    # d.plot_ts()
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # IAED Model definition
    config = init_config(cCNN, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = True,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = mCNNLSTM(df = df, config = config)
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
