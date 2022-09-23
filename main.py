from model.mIAED import mIAED
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model.AdjLR import AdjLR
from Data import Data
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) 
from parameters import *


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder + "plots"):
        os.makedirs(folder + "plots")


# Data initialization
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.downsample(10)
d.scale_data()
X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()
create_folder(MODEL_FOLDER + "/")


# Model definition
model = mIAED(config = config)
model.compile(loss='mse', optimizer = Adam(0.0005), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = True)
model.model().summary()
plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)


# Model fit
cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_FOLDER + '/', save_best_only = True)
cb_LR = AdjLR(model, 30, 0.2, 1)
model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = BATCH_SIZE, 
          epochs = 200, callbacks = [cb_checkpoints, cb_earlystop, cb_LR])

# Model evaluation
model.evaluate(x_test, y_test, BATCH_SIZE)