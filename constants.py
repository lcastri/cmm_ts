from enum import Enum
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = ROOT_DIR + "/training_result"

# Parameters definition
# N_FEATURES = 8
# LIST_FEATURES = ['d_g', 'v', 'risk', 'theta_g', 'omega', 'theta', 'g_seq', 'd_obs']
N_FEATURES = 3
LIST_FEATURES = ['X_0', 'X_1', 'X_2']


# MODELS
class Models(Enum):
    sIAED = "sIAED"
    mIAED = "mIAED"

MODELS = {
    Models.sIAED.value : "Single-output Input Attention Encoder Decoder",
    Models.mIAED.value : "Multi-output Input Attention Encoder Decoder",
}


# CAUSAL MATRICES
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

CM = np.array([[0.        , 0.86774952, 0.64580477],
               [0.        , 0.40596968, 0.        ],
               [0.        , 0.55929937, 0.        ]])


class CausalModel(Enum):
    FPCMCI = "FPCMCI"
    PCMCI = "PCMCI"
    CM = "CM"

CAUSAL_MODELS = {CausalModel.FPCMCI.value : CM_FPCMCI,
                 CausalModel.PCMCI.value : CM_PCMCI,
                 CausalModel.CM.value : CM}

