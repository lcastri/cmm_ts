import tensorflow as tf
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
from varname import nameof

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i : i + window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)


def get_sets(t, train_perc, val_perc, test_perc):
    train_len = int(len(t) * train_perc)
    val_len = int(len(t) * val_perc)
    test_len = int(len(t) * test_perc)
    return t[:train_len], t[train_len:train_len+val_len], t[train_len+val_len:]


# Download dataset
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

# Extracting temperature
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
df = df[5::6]
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
temp = df['T (degC)']
# temp.plot()
# plt.show()

# Defining train, validation, and test set
WINDOW_SIZE = 5
X_temp, y_temp = df_to_X_y(temp, WINDOW_SIZE)
X_temp.shape, y_temp.shape

TRAIN_PERC = 0.8
VAL_PERC = 0.1
TEST_PERC = 0.1
TRAIN_LEN = len(X_temp) * TRAIN_PERC
VAL_LEN = len(X_temp) * VAL_PERC
TEST_LEN = len(X_temp) * TEST_PERC

X_temp_train, X_temp_val, X_temp_test = get_sets(X_temp, TRAIN_PERC, VAL_PERC, TEST_PERC)
y_temp_train, y_temp_val, y_temp_test = get_sets(y_temp, TRAIN_PERC, VAL_PERC, TEST_PERC)
X_temp_train.shape, y_temp_train.shape, X_temp_val.shape, y_temp_val.shape, X_temp_test.shape, y_temp_test.shape

# Defining train, validation, and test set
model_temp = Sequential()
model_temp.add(InputLayer((WINDOW_SIZE, 1)))
model_temp.add(LSTM(64))
model_temp.add(Dense(8, 'relu'))
model_temp.add(Dense(1, 'linear'))

model_temp.summary()

cp_temp = ModelCheckpoint(nameof(model_temp)+'/', save_best_only=True)
model_temp.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model_temp.fit(X_temp_train, y_temp_train, validation_data=(X_temp_val, y_temp_val), epochs=10, callbacks=[cp_temp])


# Load learned model
model_temp = load_model(nameof(model_temp)+'/')

# Prediction on the train set
train_predictions = model_temp.predict(X_temp_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_temp_train})
plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])
plt.show()

# Prediction on the validation set
val_predictions = model_temp.predict(X_temp_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_temp_val})
plt.plot(val_results['Val Predictions'][50:100])
plt.plot(val_results['Actuals'][50:100])
plt.show()

# Prediction on the test set
test_predictions = model_temp.predict(X_temp_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_temp_test})
plt.plot(test_results['Test Predictions'][50:100])
plt.plot(test_results['Actuals'][50:100])
plt.show()