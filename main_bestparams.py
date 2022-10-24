import pickle
from models.utils import *
from keras.optimizers import Adam
from Data import Data
from keras.layers import *
from keras.models import *
from constants import *
import pandas as pd
from kerashypetune import KerasGridSearch
from models.utils import Words as W

# IAED import
from models.IAED.mIAED import mIAED
from models.IAED.sIAED import sIAED
from models.IAED.config import config as cIAED
# T2V import
from models.T2V.sT2VRNN import sT2VRNN
from models.T2V.config import config as cT2V


df, features = get_df(11)

# Parameters definition
MODEL = Models.sT2V
N_FUTURE = 48
N_PAST = 32
N_DELAY = 0
TRAIN_PERC = 0.7
VAL_PERC = 0.1
TEST_PERC = 0.2
MODEL_FOLDER = MODEL + "_BESTPARAM"


if MODEL == Models.sIAED:
    # Single-output data initialization
    TARGETVAR = 'd_g'
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

    # IAED Model definition
    config_grid = init_config(cIAED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                              ndelay = N_DELAY, nfeatures = N_FEATURES, features = None, initDEC = False,
                              use_att = True, use_cm = True, cm = None, cm_trainable = True, use_constraint = True, constraint = [0.1, 0.2])
    config_grid[W.ATTUNITS] = [256, 300, 512]
    config_grid[W.ENCDECUNITS] = [128, 256]
    config_grid[W.DECINIT] = [False, True]
    config_grid[W.D1UNITS] = [128, 256]
    config_grid[W.D2UNITS] = [64, 128]
    config_grid["epochs"] = 25
    config_grid["batch_size"] = [64, 128, 256, 512]

    hypermodel = lambda x: sIAED(config = x).create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), 
                                                          metrics = ['mse', 'mae', 'mape'], searchBest = True)


elif MODEL == Models.sT2V:
	# Single-output data initialization
    TARGETVAR = 'd_g'
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

    # IAED Model definition
    config_grid = init_config(cIAED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                              ndelay = N_DELAY, nfeatures = N_FEATURES, features = None, initDEC = False,
                              use_att = True, use_cm = True, cm = None, cm_trainable = True, use_constraint = True, constraint = [0.1, 0.2])
    config_grid[W.ATTUNITS] = [64, 128, 256, 512]
    config_grid[W.T2VUNITS] = [64, 128, 256, 512]
    config_grid[W.ENCDECUNITS] = [64, 128, 256, 512]
    config_grid[W.D1UNITS] = [64, 128, 256]
    config_grid[W.D2UNITS] = [32, 64, 128]
    config_grid["epochs"] = 25
    config_grid["batch_size"] = [32, 64, 128]

    hypermodel = lambda x: sT2VRNN(config = x).create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), 
                                                            metrics = ['mse', 'mae', 'mape'], searchBest = True)


elif MODEL == Models.mIAED:
    # Multi-output data initialization
    d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
    d.downsample(step = 10)
    d.smooth(window_size = 50)
    X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

    # IAED Model definition
    config_grid = init_config(cIAED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                              ndelay = N_DELAY, nfeatures = N_FEATURES, features = None, initDEC = False,
                              use_att = True, use_cm = True, cm = None, cm_trainable = True, use_constraint = True, constraint = [0.1, 0.2])
    config_grid[W.ATTUNITS] = [256, 300, 512]
    config_grid[W.ENCDECUNITS] = [128, 256]
    config_grid[W.DECINIT] = [False, True]
    config_grid[W.D1UNITS] = [128, 256]
    config_grid[W.D2UNITS] = [64, 128]
    config_grid["epochs"] = 25
    config_grid["batch_size"] = [64, 128, 256, 512]

    hypermodel = lambda x: mIAED(config = x).create_model(loss = 'mse', optimizer = Adam(0.0001),
                                                          metrics = ['mse', 'mae', 'mape'], searchBest = True)

kgs = KerasGridSearch(hypermodel, config_grid, monitor = 'val_loss', greater_is_better = False, tuner_verbose = 1)
kgs.search(X_train, y_train, validation_data = (X_val, y_val), shuffle = False)
with open(RESULT_DIR + '/' + MODEL_FOLDER + '/best_param.pkl', 'rb') as pickle_file:
        pickle.dump(kgs.best_params, pickle_file)




