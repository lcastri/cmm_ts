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
        actualY_t = np.squeeze(actualY[t,:,:])
        predY_t = np.squeeze(predY[t,:,:])
        actualY_t = d.scaler.inverse_transform(actualY_t)
        predY_t = d.scaler.inverse_transform(predY_t)
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

        # plt.figure()

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

            plt.plot(range(t, t + len(X_t[:, f_idx])), X_t[:, f_idx], color='green')
            plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(Y_t[:, f_idx])), Y_t[:, f_idx], color='blue')
            plt.plot(range(t - 1 + len(X_t[:, f_idx]), t - 1 + len(X_t[:, f_idx]) + len(predY_t[:, f_idx])), predY_t[:, f_idx], color='red')
            plt.title("Multi-step prediction - " + f)

            # plt.draw()
            plt.savefig(folder + "predictions/" + str(f) + "/" + str(t) + ".png")

            plt.clf()



def compare_causalW(mp, initial_causal_matrix):
    layers = mp['layers']
    ca_layers = [l for l in layers if l['class_name'] == "CausalAttention"]
    ca_matrix = np.array([list(ca_layers[var]['config']['causal_attention_weights'].values()) for var in range(len(ca_layers))])
    print(ca_matrix)
    print(initial_causal_matrix)


# Prepare timeseries
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.downsample(10)
d.scale_data()
_, _, _, _, X_test, y_test = d.get_timeseries()


# Load learned model
model_folder = "model_F100step_P200step_noatt/"
model = load_model(model_folder)
# plot_prediction(X_test, y_test, model_folder)
# model_params = model.get_config()
# compare_causalW(model_params, CM_FPCMCI)
# Evaluate predictions
noatt_rmse_test = evaluate(X_test, y_test)
clear_session()


# Load learned model
model_folder = "model_F100step_P200step_att/"
model = load_model(model_folder)
# plot_prediction(X_test, y_test, model_folder)
# model_params = model.get_config()
# compare_causalW(model_params, CM_FPCMCI)
# Evaluate predictions
att_rmse_test = evaluate(X_test, y_test)
clear_session()


# Load learned model
model_folder = "model_F100step_P200step_causal_fixed/"
model = load_model(model_folder)
# plot_prediction(X_test, y_test, model_folder)
# model_params = model.get_config()
# compare_causalW(model_params, CM_FPCMCI)
# Evaluate predictions
catt_rmse_test = evaluate(X_test, y_test)
clear_session()


# Load learned model
model_folder = "model_F100step_P200step_causal_train/"
model = load_model(model_folder)
# plot_prediction(X_test, y_test, model_folder)
# model_params = model.get_config()
# compare_causalW(model_params, CM_FPCMCI)
# Evaluate predictions
catttrain_rmse_test = evaluate(X_test, y_test)
clear_session()

plot_rmse(list_rmse_mean=[noatt_rmse_test, att_rmse_test, catt_rmse_test, catttrain_rmse_test], list_legends=["no-att", "att", "causal-att (fixed)","causal-att (train)"])





