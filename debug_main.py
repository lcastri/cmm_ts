from models.utils import *
from models.AdjLR import AdjLR
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from Data import Data
from keras.layers import *
from keras.models import *
from constants import *
import pandas as pd

# Models import
from models.mCNNLSTM.mIACNNED import mIACNNED
from models.mCNNLSTM.config import config as mCNNLSTM_config
from models.mIAED.mIAED import mIAED
from models.mIAED.config import config as mIAED_config
from models.mT2V.mT2VRNN import mT2VRNN
from models.mT2V.config import config as mT2V_config
from models.sIAED.sIAED import sIAED
from models.sIAED.config import config as sIAED_config
from models.sT2V.sT2VRNN import sT2VRNN
from models.sT2V.config import config as sT2V_config
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


# load csv and remove NaNs
csv_path = ROOT_DIR + "/data/training/agent_11_aug.csv"
df = pd.read_csv(csv_path)
df.fillna(method="ffill", inplace = True)
df.fillna(method="bfill", inplace = True)
features = list(df.columns)

# Parameters definition
N_FUTURE = 150
N_PAST = 20
N_DELAY = 0
TRAIN_PERC = 0.6
VAL_PERC = 0.2
TEST_PERC = 0.2
MODEL_FOLDER = "sIAED_F300_P600_catt_train"
BATCH_SIZE = 128

# # Multi-output data initialization
# d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
# d.downsample(10)
# X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

# Single-output data initialization
TARGETVAR = 'd_g'
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
d.downsample(10)
X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()


# # mIACNNED Model definition
# config = init_config(mIACNNED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
#                      use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True)
# model = mIACNNED(config = config, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])

# # mIAED Model definition
# config = init_config(mIAED_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
#                      use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True)
# model = mIAED(config = config, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])

# # mT2VRNN Model definition
# config = init_config(mT2V_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
#                      use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True)
# model = mT2VRNN(config = config, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])

# # IAED Model definition
# config = init_config(sIAED_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
#                      use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True)
# model = sIAED(config = config, target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])

# T2VRNN Model definition
config = init_config(sT2V_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                     ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
                     use_att = True, use_cm = True, cm = CM_FPCMCI, cm_trainable = True)
model = sT2VRNN(config = config, target_var = TARGETVAR, loss = 'mse', optimizer = Adam(0.0001), metrics = ['mse', 'mae', 'mape'])

# Model fit
cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_FOLDER + '/', save_best_only = True)
cb_adjLR = AdjLR(model, 2, 0.1, True, 1)
model.fit(X = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = BATCH_SIZE, 
          epochs = 300, callbacks = [cb_checkpoints, cb_earlystop, cb_adjLR])

# Model evaluation
model.RMSE(x_test, y_test, d.scaler)

# Model predictions
model.plot_predictions(x_test, y_test, d.scaler)