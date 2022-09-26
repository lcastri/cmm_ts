import os
from keras.models import load_model
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')
matplotlib.use( 'tkagg' )
from sklearn.metrics import mean_squared_error
from math import sqrt
from parameters import *
from Data import Data
import numpy as np
from keras.backend import clear_session
from tqdm import tqdm


def evaluate(X, actualY):
    # Predict and invert scaling prediction
    predY = model.predict(X)
    rmse = np.zeros(shape = (1, actualY.shape[1]))
    for t in tqdm(range(len(actualY))):
        actualY_t = np.reshape(np.squeeze(actualY[t,:,:]), newshape = (len(np.squeeze(actualY[t,:,:])), 1))
        predY_t = np.reshape(np.squeeze(predY[t,:]), newshape = (len(np.squeeze(predY[t,:])), 1))
        actualY_t = d.scalerOUT.inverse_transform(actualY_t)
        predY_t = d.scalerOUT.inverse_transform(predY_t)
        rmse = rmse + np.array([sqrt(mean_squared_error(actualY_t[f], predY_t[f])) for f in range(N_FUTURE)])
    rmse_mean = np.sum(rmse, axis=0)/len(actualY)
    return rmse_mean


def plot_rmse(list_rmse_mean, list_legends):
    plt.figure()
    plt.title("Mean RMSE vs time steps")
    for i in range(len(list_rmse_mean)):
        plt.plot(range(N_FUTURE), list_rmse_mean[i])
    plt.xlabel("Time steps")
    plt.xlabel("Mean RMSE")
    plt.legend(list_legends)
    plt.show()


def plot_prediction(X, y, folder):

    # Create prediction folder
    if not os.path.exists(folder + "predictions/"):
        os.makedirs(folder + "predictions/")

    predY = model.predict(X)
    for f in d.features:

        # Create var folder
        if not os.path.exists(folder + "predictions/" + str(f) + "/"):
            os.makedirs(folder + "predictions/" + str(f) + "/")

        f_idx = list(d.features).index(f)

        for t in tqdm(range(len(predY))):
            # test X
            X_t = np.squeeze(X[t,:,:])
            X_t = d.scaler.inverse_transform(X_t)

            # test y
            Y_t = np.squeeze(y[t,:,:])
            Y_t = d.scaler.inverse_transform(Y_t)

            # pred y
            predY_t = np.squeeze(predY[t,:,:])
            predY_t = d.scaler.inverse_transform(predY_t)

            plt.plot(range(t, t + len(X_t[:, f_idx])), X_t[:, f_idx], color = 'green', label = "past")
            plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(Y_t[:, f_idx])), Y_t[:, f_idx], color = 'blue', label = "actual")
            plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(predY_t[:, f_idx])), predY_t[:, f_idx], color = 'red', label = "pred")
            plt.title("Multi-step prediction - " + f)
            plt.xlabel("time []")
            plt.ylabel(f)
            plt.legend()
            plt.savefig(folder + "predictions/" + str(f) + "/" + str(t) + ".png")

            plt.clf()
            
    plt.close()

# def plot_prediction(X, y, folder):
#     # Create prediction folder
#     if not os.path.exists(folder + "predictions/"):
#         os.makedirs(folder + "predictions/")

#     predY = model.predict(X)

#     f = 'd_g'
#     f_idx = list(d.features).index(f)
#     # Create var folder
#     if not os.path.exists(folder + "predictions/" + str(f) + "/"):
#         os.makedirs(folder + "predictions/" + str(f) + "/")


#     for t in tqdm(range(len(predY))):
#         # test X
#         X_t = np.squeeze(X[t,:,:])
#         X_t = d.scalerIN.inverse_transform(X_t)

#         # test y
#         Y_t = np.reshape(np.squeeze(y[t,:,:]), newshape = (len(np.squeeze(y[t,:,:])), 1))
#         Y_t = d.scalerOUT.inverse_transform(Y_t)

#         # pred y
#         predY_t = np.reshape(np.squeeze(predY[t,:]), newshape = (len(np.squeeze(predY[t,:])), 1))
#         predY_t = d.scalerOUT.inverse_transform(predY_t)

#         plt.plot(range(t, t + len(X_t[:, f_idx])), X_t[:, f_idx], color='green')
#         plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(Y_t[:, f_idx])), Y_t[:, f_idx], color='blue')
#         plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(predY_t[:, f_idx])), predY_t[:, f_idx], color='red')
#         plt.title("Multi-step prediction - " + f)

#         # plt.draw()
#         plt.savefig(folder + "predictions/" + str(f) + "/" + str(t) + ".png")

#         plt.clf()


def compare_causalW(mp, initial_causal_matrix):
    layers = mp['layers']
    ca_layers = [l for l in layers if l['class_name'] == "CausalAttention"]
    ca_matrix = np.array([list(ca_layers[var]['config']['causal_attention_weights'].values()) for var in range(len(ca_layers))])
    print(ca_matrix)
    print(initial_causal_matrix)


# Prepare timeseries
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.downsample(10)
_, _, _, _, X_test, y_test = d.get_timeseries('d_g')


# Load learned model
model_folder = "model_F100step_P200step_prova2/"
model = load_model(model_folder)
plot_prediction(X_test, y_test, model_folder)
# model_params = model.get_config()
# compare_causalW(model_params, CM_FPCMCI)
# Evaluate predictions
noatt_rmse_test = evaluate(X_test, y_test)
clear_session()


# # Load learned model
# model_folder = "model_F100step_P200step_catt_train/"
# model = load_model(model_folder)
# # plot_prediction(X_test, y_test, model_folder)
# # model_params = model.get_config()
# # compare_causalW(model_params, CM_FPCMCI)
# # Evaluate predictions
# catttrain_rmse_test = evaluate(X_test, y_test)
# clear_session()

plot_rmse(list_rmse_mean=[noatt_rmse_test], list_legends=["prova"])





