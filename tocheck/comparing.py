from keras.models import load_model
from keras.backend import clear_session
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')
matplotlib.use( 'tkagg' )
from constants import *
from Data import Data


def predict(X, actualY):
    # Invert scaling actual
    actualY = d.scaler.inverse_transform(actualY)

    # Predict and invert scaling prediction
    predY = model.predict(X)
    predY = d.scaler.inverse_transform(predY)

    # Plot
    df_prediction = pd.DataFrame(predY, columns = df.columns.to_list())
    df_actual = pd.DataFrame(actualY, columns = df.columns.to_list())

    return df_prediction, df_actual

def plot_prediction(pred, actual):
    for v in df.columns.to_list():
        plt.plot(pred[v])
        plt.plot(actual[v])
        plt.title(v + " prediction")
        plt.legend(['Prediction', 'Actual'])
        plt.show()

# Prepare timeseries
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.scale_data()
X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

# Load learned model
model = load_model("model_CALSTM_nodelay_nocausal" + '/')
# model_params = model.get_config()
# Predictions
df_pred_train_nocausal, df_act_train_nocausal = predict(X_train, y_train)
df_pred_val_nocausal, df_act_val_nocausal = predict(X_val, y_val)
df_pred_test_nocausal, df_act_test_nocausal = predict(X_test, y_test)
clear_session()

# Load learned model
model = load_model("model_CALSTM_nodelay_causal_pcmci" + '/')
# model_params = model.get_config()
# Predictions
df_pred_train_pcmci, df_act_train_pcmci = predict(X_train, y_train)
df_pred_val_pcmci, df_act_val_pcmci = predict(X_val, y_val)
df_pred_test_pcmci, df_act_test_pcmci = predict(X_test, y_test)
clear_session()

# Load learned model
model = load_model("model_CALSTM_nodelay_causal_fpcmci_softmax" + '/')
# model_params = model.get_config()
# Predictions
df_pred_train_fpcmci, df_act_train_fpcmci = predict(X_train, y_train)
df_pred_val_fpcmci, df_act_val_fpcmci = predict(X_val, y_val)
df_pred_test_fpcmci, df_act_test_fpcmci = predict(X_test, y_test)
clear_session()

for v in df.columns.to_list():
    plt.plot(df_pred_train_nocausal[v])
    plt.plot(df_pred_train_pcmci[v])
    plt.plot(df_pred_train_fpcmci[v])
    plt.plot(df_act_train_nocausal[v])
    plt.title(v + " prediction - train")
    plt.legend(['pred', 'PCMCI_pred', 'FPCMCI_pred', 'actual'])
    plt.show()

for v in df.columns.to_list():
    plt.plot(df_pred_val_nocausal[v])
    plt.plot(df_pred_val_pcmci[v])
    plt.plot(df_pred_val_fpcmci[v])
    plt.plot(df_act_val_nocausal[v])
    plt.title(v + " prediction - val")
    plt.legend(['pred', 'PCMCI_pred', 'FPCMCI_pred', 'actual'])
    plt.show()

for v in df.columns.to_list():
    plt.plot(df_pred_test_nocausal[v])
    plt.plot(df_pred_test_pcmci[v])
    plt.plot(df_pred_test_fpcmci[v])
    plt.plot(df_act_test_nocausal[v])
    plt.title(v + " prediction - test")
    plt.legend(['pred', 'PCMCI_pred', 'FPCMCI_pred', 'actual'])
    plt.show()

