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
from sklearn.preprocessing import StandardScaler


def get_sets(t, train_perc, val_perc, test_perc):
    train_len = int(len(t) * train_perc)
    val_len = int(len(t) * val_perc)
    test_len = int(len(t) * test_perc)
    return t[:train_len], t[train_len:train_len+val_len], t[train_len+val_len:]


# load csv and remove NaNs
csv_path = "data/Exp_1_run_1/agent_11.csv"
df = pd.read_csv(csv_path)
df.fillna(method="ffill", inplace = True)
df.fillna(method="bfill", inplace = True)

# Scale DataFrame
df_for_training = df.astype(float)
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Parameters definition
n_future = 1
n_past = 10
TRAIN_PERC = 0.8
VAL_PERC = 0.1
TEST_PERC = 0.1

# Get timeseries ready for forecasting
X = []
Y = []
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    X.append(df_for_training_scaled[i - n_past : i, 0 : df_for_training_scaled.shape[1]])
    Y.append(df_for_training_scaled[i + n_future - 1 : i + n_future, 0 : df_for_training_scaled.shape[1]])
trainX, valX, testX = get_sets(np.array(X), TRAIN_PERC, VAL_PERC, TEST_PERC)
trainY, valY, testY = get_sets(np.array(Y), TRAIN_PERC, VAL_PERC, TEST_PERC)
print('trainX shape : {}.'.format(trainX.shape))
print('trainY shape : {}.'.format(trainY.shape))
print('valX shape : {}.'.format(valX.shape))
print('valY shape : {}.'.format(valY.shape))
print('testX shape : {}.'.format(testX.shape))
print('testY shape : {}.'.format(testY.shape))

# Define Autoencoder model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[2]))

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01), metrics=[RootMeanSquaredError()])
model.summary()

# fit model
MODEL_NAME = "model"
cp = ModelCheckpoint(MODEL_NAME + '/', save_best_only=True)
history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=300, callbacks=[cp])

plt.plot(history.history["loss"], label = "Training loss")
plt.plot(history.history["val_loss"], label = "Validation loss")
plt.legend()
plt.show()

# Load learned model
model = load_model(MODEL_NAME + '/')

#TODO: Inverse transform from scale space to normal
# scaler.inverse_transform()

# Prediction on the train set
train_predictions = model.predict(trainX)
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':trainY})
plt.plot(train_results['Train Predictions'])
plt.plot(train_results['Actuals'])
plt.show()

# Prediction on the validation set
val_predictions = model.predict(valX)
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':valY})
plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])
plt.show()

# Prediction on the test set
test_predictions = model.predict(testX)
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':testY})
plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])
plt.show()