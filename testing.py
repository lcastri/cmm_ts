import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')
matplotlib.use( 'tkagg' )
from parameters import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from Data import Data

def split_sequence(sequence, look_back, forecast_horizon):
 X, y = list(), list()
 for i in range(len(sequence)): 
   lag_end = i + look_back
   forecast_end = lag_end + forecast_horizon
   if forecast_end > len(sequence):
     break
   seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
   X.append(seq_x)
   y.append(seq_y)
 return np.array(X), np.squeeze(np.array(y))

def predict(X, actualY):
    # Invert scaling actual
    actualY = d.scaler.inverse_transform(actualY)

    # Predict and invert scaling prediction
    predY = model.predict(X)
    predY = d.scaler.inverse_transform(predY)

    # Plot
    df_prediction = pd.DataFrame(predY, columns = df.columns.to_list())
    df_actual = pd.DataFrame(actualY, columns = df.columns.to_list())
    print('RMSE')
    for v in df.columns.to_list():
        plt.plot(df_prediction[v])
        plt.plot(df_actual[v])
        plt.title(v + " prediction")
        plt.legend(['Prediction', 'Actual'])
        plt.show()
        rmse = sqrt(mean_squared_error(df_actual[v], df_prediction[v]))
        print('{:<10s}{:^5s}{:>10f}'.format(v, ":", rmse))

# Prepare timeseries
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.scale_data()
X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

# Load learned model
model = load_model(MODEL_NAME + '/')

# Predictions
predict(X_train, y_train)
predict(X_val, y_val)
predict(X_test, y_test)


