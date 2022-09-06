from matplotlib import pyplot as plt
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from parameters import *

X, y = split_sequences(df_scaled, N_PAST, N_FUTURE)
print ("X.shape" , X.shape) 
print ("y.shape" , y.shape)

trainX, valX, testX = get_sets(X, TRAIN_PERC, VAL_PERC, TEST_PERC)
trainY, valY, testY = get_sets(y, TRAIN_PERC, VAL_PERC, TEST_PERC)
print ("trainX.shape" , trainX.shape) 
print ("trainy.shape" , trainY.shape)
print ("valX.shape" , valX.shape) 
print ("valy.shape" , valY.shape)
print ("testX.shape" , testX.shape) 
print ("testy.shape" , testY.shape)

# Define model
m = Sequential()
m.add(LSTM(64, input_shape = (N_PAST, N_FEATURES)))
m.add(Dropout(0.5))
m.add(RepeatVector(N_FUTURE))
m.add(LSTM(32, return_sequences = True))
m.add(Dropout(0.5))
m.add(TimeDistributed(Dense(N_FEATURES, activation='linear')))
m.compile(loss='mse', optimizer=Adam(0.00001))
m.summary()


# fit model
cb_earlystop = EarlyStopping(monitor='loss', patience = 3)
cb_checkpoints = ModelCheckpoint(MODEL_NAME + '/', save_best_only=True)
history = m.fit(trainX, trainY, 
                validation_data = (valX, valY), 
                epochs = 30, 
                callbacks=[cb_earlystop, cb_checkpoints])

plt.plot(history.history["loss"], label = "Training loss")
plt.plot(history.history["val_loss"], label = "Validation loss")
plt.legend()
plt.show()
