from models.words import *


config = {

    W_SETTINGS : {
        W_FOLDER : None,
        W_NPAST : None,
        W_NFUTURE : None,
        W_NDELAY : None,
        W_NFEATURES : None,
        W_FEATURES : None,
        W_USEATT : False
    },

    W_INPUTATT : {
        W_USECAUSAL : False,
        W_CMATRIX : None,
        W_CTRAINABLE : False,
        W_USECONSTRAINT : False,
        W_TRAINTHRESH : 0.2
    },

    W_T2V : {
        W_UNITS : 64
    },

    W_RNN : {
        W_UNITS : 128
    },
    
    W_OUT : [
        {W_UNITS : 32,
         W_ACT : "relu"},
    ]
}