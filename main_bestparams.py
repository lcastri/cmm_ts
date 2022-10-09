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

# Models import
from models.mIAED import mIAED
from models.sIAED import sIAED
from models.config import config 

# load csv and remove NaNs
csv_path = ROOT_DIR + "/data/training/agent_11_aug.csv"
df = pd.read_csv(csv_path)
df.fillna(method = "ffill", inplace = True)
df.fillna(method = "bfill", inplace = True)
features = list(df.columns)

# Parameters definition
MODEL = Models.sIAED.value
N_FUTURE = 150
N_PAST = 20
N_DELAY = 0
TRAIN_PERC = 0.6
VAL_PERC = 0.2
TEST_PERC = 0.2
MODEL_FOLDER = MODEL + "_BESTPARAM"
BATCH_SIZE = 128

if MODEL == Models.sIAED.value:
        # Single-output data initialization
        TARGETVAR = 'd_g'
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
        d.downsample(10)
        X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

        # IAED Model definition
        config_grid = init_config(config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                                  ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = False,
                                  use_att = True, use_cm = True, cm = None, cm_trainable = True, use_constraint = True, constraint = 0.2)
        config_grid[W.ATTUNITS] = [128, 256, 300]
        config_grid[W.ENCDECUNITS] = [128, 256]
        config_grid[W.DECINIT] = [False, True]
        config_grid[W.D1UNITS] = [64, 128, 256]
        config_grid[W.D2UNITS] = [64, 128]
        config_grid["epochs"] = 50
        config_grid["batch_size"] = [64, 128, 256]

        hypermodel = lambda x: sIAED(config = x).create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), 
                                                              metrics = ['mse', 'mae', 'mape'], searchBest = True)

elif MODEL == Models.mIAED.value:
        # Multi-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
        d.downsample(10)
        X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

        # IAED Model definition
        config_grid = init_config(config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                                  ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = False,
                                  use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True, use_constraint = True, constraint = 0.2)
        config_grid[W.ATTUNITS] = [128, 256, 300]
        config_grid[W.ENCDECUNITS] = [128, 256]
        config_grid[W.DECINIT] = [False, True]
        config_grid[W.D1UNITS] = [128, 256]
        config_grid[W.D2UNITS] = [64, 128]
        config_grid["epochs"] = 50
        config_grid["batch_size"] = [128, 256, 512]

        hypermodel = lambda x: mIAED(config = x).create_model(loss = 'mse', optimizer = Adam(0.0001),
                                                              metrics = ['mse', 'mae', 'mape'], searchBest = True)


kgs = KerasGridSearch(hypermodel, config_grid, monitor = 'val_loss', greater_is_better = False, tuner_verbose = 1)
kgs.search(X_train, y_train, validation_data = (X_val, y_val), shuffle = False)
with open(RESULT_DIR + '/' + MODEL_FOLDER + '/best_param.pkl', 'rb') as pickle_file:
        pickle.dump(kgs.best_params, pickle_file)




