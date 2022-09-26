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

    W_CNN : [
        {W_FILTERS : 128,
         W_KSIZE : 4,
         W_ACT : 'tanh'},
        {W_FILTERS : 64,
         W_KSIZE : 2,
         W_ACT : 'tanh'}
        ],


    W_ENC : [
        {W_UNITS : 64,
         W_RSEQ : False,
         W_RSTATE : False,
         W_ACT : 'tanh'}
        ],

    W_DEC : [
        {W_UNITS : 64,
         W_RSEQ : True,
         W_RSTATE : False,
         W_ACT : 'tanh'}
        ],


    W_OUT : [
        # {W_UNITS : 256,
        #  W_DROPOUT : 0.5,
        #  W_ACT : "relu"},
        # {W_UNITS : 64,
        #  W_DROPOUT : None,
        #  W_ACT : "relu"},
        {W_UNITS : 32,
         W_DROPOUT : None,
         W_ACT : "relu"}
    ]
}