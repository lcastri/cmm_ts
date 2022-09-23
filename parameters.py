import pandas as pd
import numpy as np
from model.words import *

# load csv and remove NaNs
csv_path = "data/training/agent_11.csv"
df = pd.read_csv(csv_path)
df.fillna(method="ffill", inplace = True)
df.fillna(method="bfill", inplace = True)
features = list(df.columns)

# Parameters definition
N_FUTURE = 10
N_PAST = 20
N_DELAY = 0
N_FEATURES = 8
TRAIN_PERC = 0.6
VAL_PERC = 0.2
TEST_PERC = 0.2
MODEL_FOLDER = "model_F50step_P100step_causal_train4"
BATCH_SIZE = 300


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



config = {

    W_SETTINGS : {
        W_FOLDER : MODEL_FOLDER,
        W_NPAST : N_PAST,
        W_NFUTURE : N_FUTURE,
        W_NDELAY : N_DELAY,
        W_NFEATURES : N_FEATURES,
        W_FEATURES : features,
        W_USEATT : True
    },

    W_INPUTATT : {
        W_USECAUSAL : True,
        W_CMATRIX : CM_FPCMCI,
        W_CTRAINABLE : True,
        W_USECONSTRAINT : False,
        W_TRAINTHRESH : 0.2
    },

    W_ENC : [
        {W_UNITS : 32,
         W_RSEQ : True,
         W_RSTATE : True},
        {W_UNITS : 16,
         W_RSEQ  : False,
         W_RSTATE : True}
        ],

    W_DEC : [
        {W_UNITS : 32,
         W_RSEQ : True,
         W_RSTATE : False},
        {W_UNITS : 16,
         W_RSEQ : True,
         W_RSTATE : False}
        ],

    W_OUT : [
        # {W_UNITS : 256,
        #  W_DROPOUT : 0.5,
        #  W_ACT : "relu"},
        {W_UNITS : 64,
         W_DROPOUT : 0.5,
         W_ACT : "relu"},
        {W_UNITS : 32,
         W_DROPOUT : 0.5,
         W_ACT : "relu"}
    ]
}



config_2 = {

    W_SETTINGS : {
        W_FOLDER : MODEL_FOLDER,
        W_NPAST : N_PAST,
        W_NFUTURE : N_FUTURE,
        W_NDELAY : N_DELAY,
        W_NFEATURES : N_FEATURES,
        W_FEATURES : features,
        W_USEATT : False
    },

    W_INPUTATT : {
        W_USECAUSAL : False,
        W_CMATRIX : CM_FPCMCI,
        W_CTRAINABLE : False,
        W_USECONSTRAINT : False,
        W_TRAINTHRESH : 0.2
    },

    W_CNN : [
        {W_FILTERS : 10,
         W_KSIZE : 20,
         W_ACT : 'relu'},
        {W_FILTERS : 20,
         W_KSIZE : 40,
         W_ACT : 'relu'},
        ],


    W_OUT : [
        # {W_UNITS : 256,
        #  W_DROPOUT : 0.5,
        #  W_ACT : "relu"},
        {W_UNITS : 64,
         W_DROPOUT : 0.5,
         W_ACT : "relu"},
        {W_UNITS : 32,
         W_DROPOUT : 0.5,
         W_ACT : "relu"}
    ]
}