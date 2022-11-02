from Data import Data
from constants import RESULT_DIR
# IAED import
from models.IAED.mIAED import mIAED
from models.IAED.sIAED import sIAED
from models.IAED.config import config as cIAED
# T2V import
from models.T2V.mT2VRNN import mT2VRNN
from models.T2V.sT2VRNN import sT2VRNN
from models.T2V.config import CONFIG as cT2V
# CNN import
from models.CNNLSTM.mCNNLSTM import mCNNLSTM
from models.CNNLSTM.sCNNLSTM import sCNNLSTM
from models.CNNLSTM.config import CONFIG as cCNN
from models.utils import get_df


N_FUTURE = 48
N_PAST = 32
N_DELAY = 0
TRAIN_PERC = 0.5
VAL_PERC = 0.1
TEST_PERC = 0.4


TEST_AGENTS = [11]
MODELS = ["256_mIAED_FPCMCI_f"]
for M in MODELS:
    for a in TEST_AGENTS:
        df, features = get_df(a)
        m = mIAED(df = df, folder = M)
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
        d.downsample(10)
        d.smooth(window_size = 50)
        _, _, _, _, X_test, y_test = d.get_timeseries()

        # # Model evaluation
        # m.MAE(X_test, y_test, d.scaler, folder = RESULT_DIR + "/" + M + '/test/' + str(a))
        
        # Model predictions
        m.predict(X_test, y_test, d.scaler, folder = RESULT_DIR + "/" + M + '/prediction/' + str(a), plot = True)

    