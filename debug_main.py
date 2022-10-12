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
from models.config import config
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


# # load csv and remove NaNs
# csv_path = ROOT_DIR + "/data/training/agent_11_aug.csv"
# df = pd.read_csv(csv_path)
# df.fillna(method="ffill", inplace = True)
# df.fillna(method="bfill", inplace = True)
# features = list(df.columns)

df, features = get_df(3)

# Parameters definition
MODEL = Models.mIAED.value
N_FUTURE = 150
N_PAST = 20
N_DELAY = 0
TRAIN_PERC = 0.6
VAL_PERC = 0.2
TEST_PERC = 0.2
MODEL_FOLDER = "PROVA"
BATCH_SIZE = 64
PATIENCE = 25
EPOCH = 1
TARGETVAR = 'd_g'

if MODEL == Models.sIAED.value:
    if TARGETVAR == None: raise ValueError('for models sIAED, target_var needs to be specified')
    # Single-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
    d.downsample(10)
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # IAED Model definition
    config = init_config(config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = True,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = sIAED(config = config)
    model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


elif MODEL == Models.mIAED.value:
    # Multi-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
    d.downsample(10)
    X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

    # IAED Model definition
    config = init_config(config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                         ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = True,
                         use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
    model = mIAED(config = config)
    model.create_model(loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


# # Single-output data initialization
# TARGETVAR = 'd_g'
# d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
# d.downsample(10)
# X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

# # IAED Model definition
# config = init_config(config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
#                      use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True)
# model = sIAED(config = config)
# model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])

# # Multi-output data initialization
# d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
# d.downsample(10)
# X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

# # mIAED Model definition
# config = init_config(config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
#                      use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True)
# model = mIAED(config = config)
# model.create_model(loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])


# Model fit
cbs = list()
cbs.append(EarlyStopping(patience = PATIENCE))
cbs.append(ModelCheckpoint('training_result/' + MODEL_FOLDER + '/', save_best_only = True))
model.fit(X = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = BATCH_SIZE, 
          epochs = EPOCH, callbacks = cbs)

model.save_cmatrix()

# Model evaluation
model.RMSE(X_test, y_test, d.scaler)

# Model predictions
model.predict(X_test, y_test, d.scaler)
