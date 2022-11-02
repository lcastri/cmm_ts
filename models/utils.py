import os 
import logging
import tensorflow as tf
import absl.logging
from constants import *
import models.Words as Words
import pandas as pd


def init_config(config, folder, npast, nfuture, ndelay, nfeatures, features, initDEC = False,
                use_att = False, use_cm = False, cm = None, cm_trainable = False, use_constraint = False, constraint = None):
    """
    Init network configuration 

    Args:
        config (dict): empty network configuration
        folder (str): model folder
        npast (int): observation window
        nfuture (int): forecasting window
        ndelay (int): forecasting delay
        nfeatures (int): number of input variables
        features (list[str]): input variables
        initDEC (bool, optional): use encoder final state as initial state for decoder. Defaults to False.
        use_att (bool, optional): use attention mechanism. Defaults to False.
        use_cm (bool, optional): use causal model. Defaults to False.
        cm (np.array, optional): causal matrix to use. Defaults to None.
        cm_trainable (bool, optional): causal model trainable. Defaults to False.
        use_constraint (bool, optional): causal model constraint flag. Defaults to False.
        constraint (float, optional): causal model constraint. Defaults to None.

    Returns:
        dict: network configuration
    """
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
    """
    Disable warning
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.get_logger().setLevel(logging.ERROR)
    absl.logging.set_verbosity(absl.logging.ERROR) 


def cmd_attention_map(att, catt):
    """
    Convert input from parser to boolean values

    Args:
        att (bool): --att parser option
        catt ((str, bool, float)): --catt parser option
    """
    def strTrue(s): return s == 'True'
    def strNone(s): return s == 'None' or s is None

    cm = CAUSAL_MODELS[catt[0]] if catt[0] is not None else None
    cm_trainable = strTrue(catt[1])
    constraint = float(catt[2]) if not strNone(catt[2]) else None
    
    use_cm = cm is not None
    use_constraint = constraint is not None

    use_att = att or use_cm

    return use_att, use_cm, cm, cm_trainable, use_constraint, constraint


def get_df(csv):
    """
    load csv and remove NaNs

    Args:
        csv (str): path fo file.csv 

    Returns:
        Dataframe: loaded dataframe
        list[str]: dataframe var names
    """
    csv_path = ROOT_DIR + "/data/" + str(csv)
    df = pd.read_csv(csv_path)

    df.fillna(method="ffill", inplace = True)
    df.fillna(method="bfill", inplace = True)
    features = list(df.columns)
    return df, features
