import glob
import os 
import logging
import tensorflow as tf
import absl.logging
from constants import *
from enum import Enum
import models.Words as Words
import pandas as pd
from natsort import natsorted


class Models(Enum):
    sIAED = "sIAED"
    mIAED = "mIAED"


class CausalModel(Enum):
    FPCMCI = "FPCMCI"
    PCMCI = "PCMCI"


CAUSAL_MODELS = {CausalModel.FPCMCI.value : CM_FPCMCI,
                 CausalModel.PCMCI.value : CM_PCMCI}


MODELS = {
    Models.sIAED.value : "Single-output Input Attention Encoder Decoder",
    Models.mIAED.value : "Multi-output Input Attention Encoder Decoder",
}


def init_config(config, folder, npast, nfuture, ndelay, nfeatures, features, initDEC = False,
                use_att = False, use_cm = False, cm = None, cm_trainable = False, use_constraint = False, constraint = None):
    config[Words.FOLDER] = folder
    config[Words.NPAST] = npast
    config[Words.NFUTURE] = nfuture
    config[Words.NDELAY] = ndelay
    config[Words.NFEATURES] = nfeatures
    config[Words.FEATURES] = features
    config[Words.USEATT] = use_att
    config[Words.USECAUSAL] = use_cm
    config[Words.CMATRIX] = cm
    config[Words.CTRAINABLE] = cm_trainable
    config[Words.USECONSTRAINT] = use_constraint
    config[Words.TRAINTHRESH] = constraint
    config[Words.DECINIT] = initDEC
    return config


def no_warning():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.get_logger().setLevel(logging.ERROR)
    absl.logging.set_verbosity(absl.logging.ERROR) 


def cmd_attention_map(att, catt):
    def strTrue(s): return s == 'True'
    def strNone(s): return s == 'None' or s is None

    cm = CAUSAL_MODELS[catt[0]] if catt[0] is not None else None
    cm_trainable = strTrue(catt[1])
    constraint = float(catt[2]) if not strNone(catt[2]) else None
    
    use_cm = cm is not None
    use_constraint = constraint is not None

    use_att = att or use_cm

    return use_att, use_cm, cm, cm_trainable, use_constraint, constraint


def get_df(agent):
    # load csv and remove NaNs
    csv_path = ROOT_DIR + "/data/" + str(agent) + "/"

    all_files = natsorted(glob.glob(os.path.join(csv_path, "*.csv")))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index = True)

    df.fillna(method="ffill", inplace = True)
    df.fillna(method="bfill", inplace = True)
    features = list(df.columns)
    return df, features
