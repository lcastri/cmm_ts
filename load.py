from Data import Data
from constants import RESULT_DIR
from models.IAED.mIAED import mIAED
from models.utils import get_df


N_FUTURE = 150
N_PAST = 20
N_DELAY = 0
TRAIN_PERC = 0.0
VAL_PERC = 0.0
TEST_PERC = 1.0


TEST_AGENTS = [3, 4, 5, 6, 7, 8, 9, 10]
MODELS = ["mIAED", "mIAED_att", "FPCMCI_mIAED_catt_tc", "PCMCI_mIAED_catt_tc"]
for M in MODELS:
    m = mIAED(folder = M)
    for a in TEST_AGENTS:
        df, features = get_df(a)
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
        d.downsample(step = 10)
        d.smooth(window_size = 50)
        _, _, _, _, X_test, y_test = d.get_timeseries()

        # Model evaluation
        m.RMSE(X_test, y_test, d.scaler, folder = RESULT_DIR + "/" + M + '/test/' + str(a))
        # # Model predictions
        # m.predict(X_test, y_test, d.scaler, folder = RESULT_DIR + "/" + M + '/prediction/' + str(a))

    



# m.mean_RMSE(np.load(m.model_dir +  "/rmse.npy"))

# model.save_cmatrix()

# # Model evaluation
# model.RMSE(x_test, y_test, d.scaler)

# # Model predictionsmk dir 
# model.plot_predictions(x_test, y_test, d.scaler)