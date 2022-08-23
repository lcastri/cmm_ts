from locale import D_FMT
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


def preprocess(x):
    for f in range(x.shape[-1]):
        x[:,:,f] =  (x[:,:,f] - np.mean(x[:,:,f])) / np.std(x[:,:,f])
    return x


def postprocess(x):
    for f in range(x.shape[-1]):
        x[:,:,f] =  x[:,:,f] * np.std(x[:,:,f]) + np.mean(x[:,:,f])
    return x

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [a for a in df_as_np[i : i + window_size]]
        X.append(row)
        label = [y for y in df_as_np[i + window_size]]
        y.append(label)
    return np.array(X), np.array(y)


def get_sets(t, train_perc, val_perc, test_perc):
    train_len = int(len(t) * train_perc)
    val_len = int(len(t) * val_perc)
    test_len = int(len(t) * test_perc)
    return t[:train_len], t[train_len:train_len+val_len], t[train_len+val_len:]


# Extracting temperature
csv_path = "data/Exp_1_run_1/agent_11.csv"
df = pd.read_csv(csv_path)

# Defining train, validation, and test set
N = len(df.columns)
WINDOW_SIZE = 5
MODEL_NAME = "model/" + csv_path[5:-4]
X, y = df_to_X_y(df, WINDOW_SIZE)
X.shape, y.shape

TRAIN_PERC = 0.8
VAL_PERC = 0.1
TEST_PERC = 0.1

X_train, X_val, X_test = get_sets(X, TRAIN_PERC, VAL_PERC, TEST_PERC)
y_train, y_val, y_test = get_sets(y, TRAIN_PERC, VAL_PERC, TEST_PERC)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

X_train = preprocess(X_train)
X_val = preprocess(X_val)
X_test = preprocess(X_test)
y_train = preprocess(y_train)
y_val = preprocess(y_val)
y_test = preprocess(y_test)

# Defining train, validation, and test set
model_temp = Sequential()
model_temp.add(InputLayer((WINDOW_SIZE, N)))
model_temp.add(LSTM(128))
model_temp.add(Dense(64, 'relu'))
model_temp.add(Dense(N, 'linear'))

model_temp.summary()

cp_temp = ModelCheckpoint(MODEL_NAME + '/', save_best_only=True)
model_temp.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model_temp.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp_temp])

# Load learned model
model_temp = load_model(MODEL_NAME + '/')

# Prediction on the train set
train_predictions = model_temp.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])
plt.show()

# Prediction on the validation set
val_predictions = model_temp.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])
plt.show()

# Prediction on the test set
test_predictions = model_temp.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])
plt.show()