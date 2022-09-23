from keras.models import load_model
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')
matplotlib.use( 'tkagg' )
from parameters import *
from Data import Data


def predict(X, actualY):
    # Invert scaling actual
    actualY = d.scaler.inverse_transform(np.squeeze(actualY))

    # Predict and invert scaling prediction
    predY = np.squeeze(model.predict(X))
    predY = d.scaler.inverse_transform(predY)

    # Plot
    df_prediction = pd.DataFrame(predY, columns = df.columns.to_list())
    df_actual = pd.DataFrame(actualY, columns = df.columns.to_list())

    return df_prediction, df_actual

def plot_prediction(pred, actual, label):
    for v in df.columns.to_list():
        plt.figure()
        plt.plot(pred[v])
        plt.plot(actual[v])
        plt.title(v + " prediction - " + label)
        plt.legend(['Prediction', 'Actual'])
    plt.show()

# Prepare timeseries
d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
d.downsample(10)
d.scale_data()
_, _, _, _, X_test, y_test = d.get_timeseries()

# Load learned model
model = load_model("model_CALSTM_causal_P20_F10" + '/')
model_params = model.get_config()
# Predictions
df_pred_test, df_act_test = predict(X_test, y_test)
# Plot predictions
plot_prediction(df_pred_test, df_act_test, "test")



