import os
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from Data import Data
import pandas as pd
from constants import ROOT_DIR
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

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
MODEL_FOLDER = "PROVA"
BATCH_SIZE = 128

# Single-output data initialization
TARGETVAR = 'd_g'
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
d.downsample(10)
X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()



print(tf.executing_eagerly())
# Define an input sequence and process it.
encoder_inputs = Input(shape=(20, 8))
encoder = LSTM(128, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
print("encoder_outputs", encoder_outputs[0].numpy())
print("state_h", state_h[0].numpy())
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(150, 1))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(128, return_sequences=True, return_state=False)
decoder_outputs = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense1 = Dense(64, activation='relu')
decoder_dense2 = Dense(32, activation='relu')
decoder_dense3 = Dense(1, activation='softmax')
y = decoder_dense1(decoder_outputs)
y = decoder_dense2(y)
y = decoder_dense3(y)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], y)


if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Model fit
cb_earlystop = EarlyStopping(patience = 10)
cb_checkpoints = ModelCheckpoint(MODEL_FOLDER + '/', save_best_only = True)
# Run training
model.compile(optimizer='adam', loss='mse', run_eagerly = True)
model.summary()
plot_model(model, to_file = MODEL_FOLDER + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)
model.fit([X_train, y_train], y_train,
          batch_size = BATCH_SIZE,
          epochs = 10,
          validation_split=0.2)