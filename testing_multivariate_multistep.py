from keras.models import load_model
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')
matplotlib.use( 'tkagg' )
from sklearn.metrics import mean_squared_error
from math import sqrt
from parameters import *
import numpy as np

def eval(X, actualY):
    predictionY = model.predict(X)
    rmse = list()
    for b in range(len(predictionY)):
        actualY[b,:,:] = scaler.inverse_transform(actualY[b,:,:])
        predictionY[b,:,:] = scaler.inverse_transform(predictionY[b,:,:])
        tmp_rmse = [sqrt(mean_squared_error(actualY[b,f,:], predictionY[b,f,:])) for f in range(N_FUTURE)]
        rmse.append(tmp_rmse)
    rmse = np.array(rmse)
    rmse_mean = rmse.sum(axis=0)
    rmse_mean = rmse_mean/len(rmse)
    for i in range(len(rmse_mean)):
        print('t+%d RMSE: %f' % ((i+1), rmse_mean[i]))
    
    plt.plot(range(N_FUTURE) , rmse_mean)
    plt.title("RMSE prediction")
    plt.show()

def predict(X, actualY):
    predictionY = model.predict(X)

    gt = np.append(X, actualY, axis = 1)
    prediction = np.append(X, predictionY, axis = 1)

    for b in range(len(prediction)):
        gt[b,:,:] = scaler.inverse_transform(gt[b,:,:])
        prediction[b,:,:] = scaler.inverse_transform(prediction[b,:,:])

        df_prediction = pd.DataFrame(prediction[b,:,:], columns = df.columns.to_list())
        df_actual = pd.DataFrame(gt[b,:,:], columns = df.columns.to_list())
        print('RMSE')
        for v in df.columns.to_list():
            plt.plot(df_prediction[v])
            plt.plot(df_actual[v])
            plt.title(v + " prediction")
            plt.legend(['Prediction', 'Actual'])
            plt.show()
            rmse = sqrt(mean_squared_error(df_actual[v], df_prediction[v]))
            print('{:<10s}{:^5s}{:>10f}'.format(v, ":", rmse))


# Load learned model
model = load_model(MODEL_NAME + '/')

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

# Predictions
# predict(trainX, trainY)
# predict(valX, valY)
# predict(testX, testY)

eval(trainX, trainY)





