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
from models.mIAED import mIAED
from models.sIAED import sIAED
from models.config import config
from MyParser import *


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
    INITDEC = args.noinit_dec
    BATCH_SIZE = args.batch_size
    TRAIN_PERC, VAL_PERC, TEST_PERC = args.percs
    PATIENCE = args.patience
    EPOCHS = args.epochs
    LR = args.learning_rate
    ADJLR = args.adjLR
    TARGETVAR = args.target_var

    use_att, use_cm, cm, cm_trainable, use_constraint, constraint = cmd_attention_map(args.att, args.catt)

    if MODEL == Models.sIAED.value:
        if TARGETVAR == None: raise ValueError('for models sIAED, target_var needs to be specified')
        # Single-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
        d.downsample(10)
        X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = INITDEC,
                             use_att = use_att, use_cm = use_cm, cm = cm, cm_trainable = cm_trainable, use_constraint = use_constraint, constraint = constraint)
        model = sIAED(config = config)
        model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])


    elif MODEL == Models.mIAED.value:
        # Multi-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
        d.downsample(10)
        X_train, y_train, X_val, y_val, x_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(config, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = INITDEC,
                             use_att = use_att, use_cm = use_cm, cm = cm, cm_trainable = cm_trainable, use_constraint = use_constraint)
        model = mIAED(config = config)
        model.create_model(loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])

    # Create .txt file with model parameters
    print_init(MODEL, TARGETVAR, MODEL_FOLDER, N_PAST, N_FUTURE, N_DELAY, INITDEC, TRAIN_PERC, VAL_PERC, TEST_PERC,
               use_att, use_cm, cm, cm_trainable, use_constraint, constraint , BATCH_SIZE, PATIENCE, EPOCHS, LR, ADJLR)

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
    model.predict(x_test, y_test, d.scaler, plot = True)