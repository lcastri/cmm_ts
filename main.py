
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
from models.utils import init_config

from keras.callbacks import ModelCheckpoint, EarlyStopping
from Data import Data
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from parameters import *
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


# # Multi-output data initialization
# d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
# d.downsample(10)
# X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

# Single-output data initialization
target_var = 'd_g'
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = target_var)
d.downsample(10)
X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()


# # mIACNNED Model definition
# config = init_config(mIACNNED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features)
# model = mIACNNED(config = config)
# model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = True)
# model.model().summary()
# plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# mIAED Model definition
# config = init_config(mIAED_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features)
# model = mIAED(config = config)
# model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = False)
# model.model().summary()
# plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# # mT2VRNN Model definition
# config = init_config(mT2V_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features)
# model = mT2VRNN(config = config)
# model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = True)
# model.model().summary()
# plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# IAED Model definition
config = init_config(sIAED_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                     ndelay = N_DELAY, nfeatures = N_FEATURES, features = features)
model = sIAED(config = config, target_var = target_var)

# # T2VRNN Model definition
# config = init_config(sT2V_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
#                      ndelay = N_DELAY, nfeatures = N_FEATURES, features = features)
# model = sT2VRNN(config = config, target_var = target_var)

# Model fit
cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_FOLDER + '/', save_best_only = True)
model.fit(X = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = BATCH_SIZE, 
                 epochs = 200, callbacks = [cb_checkpoints, cb_earlystop])

# Model evaluation
model.RMSE(x_test, y_test, d.scaler)

# Model predictions
model.plot_predictions(x_test, y_test, d.scaler)