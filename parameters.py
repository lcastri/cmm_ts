import pandas as pd
import numpy as np
from models.words import *
from models.mCNNLSTM.config import config as mCNNLSTM_config
from models.mIAED.config import config as mIAED_config
from models.mT2V.config import config as mT2V_config
from models.sIAED.config import config as sIAED_config
from models.sT2V.config import config as sT2V_config
from models.utils import init_config

# load csv and remove NaNs
csv_path = "data/training/agent_11_aug.csv"
df = pd.read_csv(csv_path)
df.fillna(method="ffill", inplace = True)
df.fillna(method="bfill", inplace = True)
features = list(df.columns)

# Parameters definition
N_FUTURE = 100
N_PAST = 200
N_DELAY = 0
N_FEATURES = 8
TRAIN_PERC = 0.6
VAL_PERC = 0.2
TEST_PERC = 0.2
MODEL_FOLDER = "sT2V_F100_P200_noatt"
BATCH_SIZE = 128


CM_PCMCI = np.array([[0.632,0.065,0.125,0.088,0.138,0.108,0.06,0.048], 
               [0.078,0.274,0.092,0.094,0.103,0.08,0.068,0.049], 
               [0.09,0.196,0.27,0.106,0.137,0.111,0.065,0.049], 
               [0.1,0.072,0.122,0.166,0.154,0.102,0.059,0], 
               [0.111,0.059,0.095,0.087,0.131,0.132,0.054,0.057], 
               [0.122,0.067,0.083,0.112,0.418,0.541,0,0], 
               [0.086,0,0.063,0.062,0.13,0.076,0,0.052], 
               [0,0,0.074,0.067,0.05,0,0.051,0.708]])


CM_FPCMCI = np.array([[0.794693885975173,0.0797596212634794,0,0,0.207147494884196,0,0,0],
               [0,0.547118275252963,0.118972264307896,0,0,0,0,0],
               [0,0.411013790019703,0.398058466007042,0,0,0,0,0.0622522550249479],
               [0,0,0,0.400374937278639,0.147993676497357,0,0.0963358126783955,0.120359691147529],
               [0,0,0,0,0.116815916574682,0,0,0],
               [0,0,0,0,0.239283926475173,0.60894990342525,0,0],
               [0,0,0,0,0,0,0.991103983176182,0],
               [0,0.0684969047836938,0.0634190046412317,0,0,0,0,0.972004088748988]])


config = init_config(sT2V_config,
                     folder = MODEL_FOLDER, 
                     npast = N_PAST,
                     nfuture = N_FUTURE,
                     ndelay = N_DELAY,
                     nfeatures = N_FEATURES,
                     features = features)

# config_T2V = {

#     W_SETTINGS : {
#         W_FOLDER : MODEL_FOLDER,
#         W_NPAST : N_PAST,
#         W_NFUTURE : N_FUTURE,
#         W_NDELAY : N_DELAY,
#         W_NFEATURES : N_FEATURES,
#         W_FEATURES : features,
#         W_USEATT : False
#     },

#     W_INPUTATT : {
#         W_USECAUSAL : False,
#         W_CMATRIX : CM_FPCMCI,
#         W_CTRAINABLE : False,
#         W_USECONSTRAINT : False,
#         W_TRAINTHRESH : 0.2
#     },

#     W_T2V : {
#         W_UNITS : 64
#     },

#     W_RNN : {
#         W_UNITS : 128
#     },
    
#     W_OUT : [
#         {W_UNITS : 32,
#          W_ACT : "relu"},
#     ]
# }