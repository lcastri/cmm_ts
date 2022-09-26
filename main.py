from T2V_model.mT2VRNN import mT2VRNN
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
from kerashypetune import KerasGridSearch


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
model = mT2VRNN(config = config)
model.compile(loss='mse', optimizer = Adam(0.00001), metrics=['mse', 'mae', 'mape', 'accuracy'], run_eagerly = True)
model.model().summary()
# plot_model(model.model(), to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)

# Model fit
cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_FOLDER + '/', save_best_only = True)
model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = BATCH_SIZE, 
          epochs = 200, callbacks = [cb_checkpoints, cb_earlystop])

# Model evaluation
model.evaluate(x_test, y_test, BATCH_SIZE)