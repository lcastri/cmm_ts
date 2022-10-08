import pickle
from models.utils import *
from models.AdjLR import AdjLR
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from Data import Data
from keras.layers import *
from keras.models import *
from constants import *
import pandas as pd
from kerashypetune import KerasGridSearch

# Models import
from models.mCNNLSTM.mIACNNED import mIACNNED
from models.mCNNLSTM.config import config as mCNNLSTM_config
from models.mIAED.mIAED import mIAED
from models.mIAED.config import config as mIAED_config
from models.mT2V.mT2VRNN import mT2VRNN
from models.mT2V.config import config as mT2V_config
from models.sIAED.sIAED_cbias_ksearch import sIAED
from models.sIAED.config import config as sIAED_config
from models.sT2V.sT2VRNN import sT2VRNN
from models.sT2V.config import config as sT2V_config


# load csv and remove NaNs
csv_path = ROOT_DIR + "/data/training/agent_11_aug.csv"
df = pd.read_csv(csv_path)
df.fillna(method = "ffill", inplace = True)
df.fillna(method = "bfill", inplace = True)
features = list(df.columns)

# Parameters definition
N_FUTURE = 150
N_PAST = 20
N_DELAY = 0
TRAIN_PERC = 0.6
VAL_PERC = 0.2
TEST_PERC = 0.2
MODEL_FOLDER = "BESTPARAM"
BATCH_SIZE = 128

# Single-output data initialization
TARGETVAR = 'd_g'
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
d.downsample(10)
X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

# # IAED Model definition
# config = init_config(sIAED_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
#                      use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = False)
# model = sIAED(config = config)
# model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])

config_grid = {
        W_FOLDER : MODEL_FOLDER,
        W_NPAST : N_PAST,
        W_NFUTURE : N_FUTURE,
        W_NDELAY : N_DELAY,
        W_NFEATURES : N_FEATURES,
        W_FEATURES : features,
        W_USEATT : True,
        W_USECAUSAL : True,
        W_CTRAINABLE : False,
        W_USECONSTRAINT : False,
        W_TRAINTHRESH : None,
        "ATTUNITS" : [128, 256, 300],
        "ENCDECUNITS" : [128, 256],
        "DECINIT" : [True, False],
        "D1UNITS" : [64, 128, 256],
        "D1ACT" : "relu",
        "D2UNITS" : [64, 128, 256],
        "D2ACT" : "relu",
        'epochs': 50,
        'batch_size': [128, 256, 512]
}


hypermodel = lambda x: sIAED(config = x).create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])

kgs_t2v = KerasGridSearch(hypermodel, config_grid, monitor='val_loss', greater_is_better = False, tuner_verbose = 1)
best_param = kgs_t2v.search(X_train, y_train, validation_data = (X_val, y_val), shuffle = False)
with open('training_result/best_param.pkl', 'rb') as pickle_file:
    pickle.dump(best_param, pickle_file)
