from models.words import *

def init_config(folder, npast, nfuture, ndelay, nfeatures, features, use_att = False, use_cm = False, cm = None, cm_trainable = False):
    config[W_SETTINGS][W_FOLDER] = folder
    config[W_SETTINGS][W_NPAST] = npast
    config[W_SETTINGS][W_NFUTURE] = nfuture
    config[W_SETTINGS][W_NDELAY] = ndelay
    config[W_SETTINGS][W_NFEATURES] = nfeatures
    config[W_SETTINGS][W_FEATURES] = features
    config[W_SETTINGS][W_USEATT] = use_att
    config[W_INPUTATT][W_USECAUSAL] = use_cm
    config[W_INPUTATT][W_CMATRIX] = cm
    config[W_INPUTATT][W_CTRAINABLE] = cm_trainable
    return config

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