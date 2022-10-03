from models.utils import *
from models.AdjLR import AdjLR
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from Data import Data
from keras.layers import *
from keras.models import *
from constants import *
import pandas as pd

# Models import
from models.mCNNLSTM.mIACNNED import mIACNNED
from models.mCNNLSTM.config import config as mCNNLSTM_config
from models.mIAED.mIAED import mIAED
from models.mIAED.config import config as mIAED_config
from models.mT2V.mT2VRNN import mT2VRNN
from models.mT2V.config import config as mT2V_config
from models.sIAED.sIAED import sIAED
from models.sIAED.config import config as sIAED_config
from models.sT2V.sT2VRNN import sT2VRNN
from models.sT2V.config import config as sT2V_config
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

import argparse
from argparse import RawTextHelpFormatter


def save_init():
    f = open(RESULT_DIR + "/" + MODEL_FOLDER + "/parameters.txt", "w")
    f.write("#\n")
    f.write("# MODEL PARAMETERS\n")
    f.write("# model = " + str(MODEL) + "\n")
    if MODEL == Models.sT2V.value or MODEL == Models.sIAED.value: f.write("# target var = " + str(TARGETVAR) + "\n")
    f.write("# model folder = " + str(MODEL_FOLDER) + "\n")
    f.write("# past window steps = " + str(N_PAST) + "\n")
    f.write("# future window steps = " + str(N_FUTURE) + "\n")
    f.write("# delay steps = " + str(N_DELAY) + "\n")
    f.write("# dataset split (train, val, test) = " + str((TRAIN_PERC, VAL_PERC, TEST_PERC)) + "\n")
    
    f.write("#" + "\n")
    f.write("# ATTENTION PARAMETERS" + "\n")
    f.write("# attention = " + str(use_att) + "\n")
    if use_att and use_cm and cm_trainable:
        if use_constraint:
            f.write("# trainable w/ contraint causality = " + str(use_cm) + "\n")
            f.write("# contraint = " + str(constraint) + "\n")
        else:
            f.write("# trainable causality = " + str(use_cm) + "\n")
    elif use_att and use_cm and not cm_trainable:
        f.write("# Fixed causality = " + str(use_cm) + "\n")
    
    f.write("#" + "\n")
    f.write("# TRAINING PARAMETERS" + "\n")
    f.write("# batch size = " + str(BATCH_SIZE) + "\n")
    f.write("# early stopping patience = " + str(PATIENCE) + "\n")
    f.write("# epochs = " + str(EPOCHS) + "\n")
    f.write("# learning rate = " + str(LR) + "\n")
    f.write("# adjust learning rate = " + str(ADJLR) + "\n")
    f.write("#\n")
    f.close()


def print_init():
    print("\n#")
    print("# MODEL PARAMETERS")
    print("# model =", MODEL)
    if MODEL == Models.sT2V.value or MODEL == Models.sIAED.value: print("# target var =", TARGETVAR)
    print("# model folder =", MODEL_FOLDER)
    print("# past window steps =", N_PAST)
    print("# future window steps =", N_FUTURE)
    print("# delay steps =", N_DELAY)
    print("# dataset split (train, val, test) =", (TRAIN_PERC, VAL_PERC, TEST_PERC))
    
    print("#")
    print("# ATTENTION PARAMETERS")
    print("# attention =", use_att)
    if use_att and use_cm and cm_trainable:
        if use_constraint:
            print("# trainable w/ contraint causality =", use_cm)
            print("# contraint =", constraint)
        else:
            print("# trainable causality =", use_cm)
    elif use_att and use_cm and not cm_trainable:
        print("# Fixed causality =", use_cm)
    
    print("#")
    print("# TRAINING PARAMETERS")
    print("# batch size =", BATCH_SIZE)
    print("# early stopping patience =", PATIENCE)
    print("# epochs =", EPOCHS)
    print("# learning rate =", LR)
    print("# adjust learning rate =", ADJLR)
    print("#\n")


def create_parser():
    model_description = "\n".join([k + " - " + MODELS[k] for k in MODELS])

    parser = argparse.ArgumentParser(description = 'Multivariate Multistep Timeseries forecasting framework.', formatter_class = RawTextHelpFormatter)
    parser.add_argument("model", type = str, choices = list(MODELS.keys()), help = model_description)
    parser.add_argument("model_dir", type = str, help = "model folder")
    parser.add_argument("--npast", type = int, help = "observation window", required = True)
    parser.add_argument("--nfuture", type = int, help = "forecasting window", required = True)
    parser.add_argument("--ndelay", type = int, help = "forecasting delay [default 0]", required = False, default = 0)
    parser.add_argument("--att", action='store_true', help = "use attention bit [default False]", required = False, default = False)
    parser.add_argument("--catt_f", action='store_true', help = "use causal-attention [FIXED] bit [default False]", required = False, default = False)
    parser.add_argument("--catt_t", action='store_true', help = "use causal-attention [TRAIN] bit [default False]", required = False, default = False)
    parser.add_argument("--catt_tc", nargs = 2, help = "use causal-attention [TRAIN w/constraint] bit [default False None]", required = False, default = [False, None])
    parser.add_argument("--target_var", type = str, help = "Target variable to forecast [used only if model = sIAED/sT2V] [default None]", required = False, default = None)
    parser.add_argument("--percs", nargs = 3, action='append', help = "[train, val, test[] percentages [default [0.6, 0.2, 0.2]]", required = False, default = [0.6, 0.2, 0.2])
    parser.add_argument("--patience", type = int, help = "earlystopping patience [default 10]", required = False, default = 10)
    parser.add_argument("--batch_size", type = int, help = "batch size [default 128]", required = False, default = 128)
    parser.add_argument("--epochs", type = int, help = "epochs [default 300]", required = False, default = 300)
    parser.add_argument("--learning_rate", type = float, help = "learning rate [default 0.0001]", required = False, default = 0.0001)
    parser.add_argument("--adjLR", nargs=3, help = "Modifying learning rate strategy (freq[epochs], factor, justOnce)", required = False)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # load csv and remove NaNs
    csv_path = ROOT_DIR + "/data/training/agent_11_aug.csv"
    df = pd.read_csv(csv_path)
    df.fillna(method="ffill", inplace = True)
    df.fillna(method="bfill", inplace = True)
    features = list(df.columns)

    MODEL = args.model
    MODEL_FOLDER = args.model_dir
    N_PAST = args.npast
    N_FUTURE = args.nfuture
    N_DELAY = args.ndelay
    BATCH_SIZE = args.batch_size
    TRAIN_PERC, VAL_PERC, TEST_PERC = args.percs
    PATIENCE = args.patience
    EPOCHS = args.epochs
    LR = args.learning_rate
    ADJLR = args.adjLR
    TARGETVAR = args.target_var
    
    use_att, use_cm, cm_trainable, use_constraint, constraint = cmd_attention_map(args.att, args.catt_f, args.catt_t, args.catt_tc)
    print_init()

    if MODEL == Models.sIAED.value:
        if TARGETVAR == None: raise ValueError('for models sIAED/sT2V, target_var needs to be specified')
        # Single-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
        d.downsample(10)
        X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(sIAED_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
                             use_att = use_att, use_cm = use_cm, cm = CM_FPCMCI, cm_trainable = cm_trainable, use_constraint = use_constraint, constraint = constraint)
        model = sIAED(config = config, target_var = TARGETVAR, loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])

    elif MODEL == Models.sT2V.value:
        if TARGETVAR == None: raise ValueError('for models sIAED/sT2V, target_var needs to be specified')
        # Single-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
        d.downsample(10)
        X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(sT2V_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
                             use_att = use_att, use_cm = use_cm, cm = CM_FPCMCI, cm_trainable = cm_trainable, use_constraint = use_constraint)
        model = sT2VRNN(config = config, target_var = TARGETVAR, loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])

    elif MODEL == Models.mIAED.value:
        # Multi-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
        d.downsample(10)
        X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(mIAED_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
                             use_att = use_att, use_cm = use_cm, cm = CM_FPCMCI, cm_trainable = cm_trainable, use_constraint = use_constraint)
        model = mIAED(config = config, loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])

    elif MODEL == Models.mT2V.value:
        # Multi-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
        d.downsample(10)
        X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(mT2V_config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features,
                             use_att = use_att, use_cm = use_cm, cm = CM_FPCMCI, cm_trainable = cm_trainable, use_constraint = use_constraint)
        model = mT2VRNN(config = config, loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])

    # Create .txt file with model parameters
    save_init()

    # Model fit
    cbs = list()
    cbs.append(EarlyStopping(patience = PATIENCE))
    cbs.append(ModelCheckpoint(RESULT_DIR + '/' + MODEL_FOLDER + '/', save_best_only = True))
    if ADJLR is not None: cbs.append(AdjLR(model, int(ADJLR[0]), float(ADJLR[1]), bool(ADJLR[2]), 1))
    model.fit(X = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = BATCH_SIZE, 
             epochs = EPOCHS, callbacks = cbs)

    # Save causal matrix
    model.save_cmatrix()

    # Model evaluation
    model.RMSE(x_test, y_test, d.scaler)

    # Model predictions
    model.plot_predictions(x_test, y_test, d.scaler)