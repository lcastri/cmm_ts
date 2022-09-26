import os 
import logging
import tensorflow as tf
import absl.logging
from words import *

def init_config(config, folder, npast, nfuture, ndelay, nfeatures, features, use_att = False, use_cm = False, cm = None, cm_trainable = False):
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



def no_warning():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.get_logger().setLevel(logging.ERROR)
    absl.logging.set_verbosity(absl.logging.ERROR) 


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder + "/plots"):
        os.makedirs(folder + "/plots")