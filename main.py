
# Models import
from models.mCNNLSTM.mIACNNED import mIACNNED
from models.mIAED.mIAED import mIAED
from models.mT2V.mT2VRNN import mT2VRNN
from models.sIAED.IAED import IAED
from models.sT2V.T2VRNN import T2VRNN

from keras.callbacks import ModelCheckpoint, EarlyStopping
from Data import Data
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from parameters import *


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
# model = mIACNNED(config = config)
# model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = True)
# model.model().summary()
# plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# # mIAED Model definition
# model = mIAED(config = config)
# model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = True)
# model.model().summary()
# plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# # mT2VRNN Model definition
# model = mT2VRNN(config = config)
# model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = True)
# model.model().summary()
# plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# # IAED Model definition
# model = IAED(config = config, target_var = target_var)
# model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = True)
# model.model().summary()
# plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# T2VRNN Model definition
model = T2VRNN(config = config, target_var = target_var)
model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse'])#, 'mae', 'mape', 'accuracy'], run_eagerly = True)
model.model().summary()
plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# Model fit
cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_FOLDER + '/', save_best_only = True)
model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = BATCH_SIZE, 
          epochs = 200, callbacks = [cb_checkpoints, cb_earlystop])

# Model evaluation
model.RMSE(x_test, y_test, d.scalerOUT)

# Model predictions
model.plot_predictions(x_test, y_test, d.scalerIN, d.scalerOUT)