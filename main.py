from models.utils import *
from models.AdjLR import AdjLR
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from Data import Data
from keras.layers import *
from keras.models import *
from constants import *

# IAED import
from models.IAED.mIAED import mIAED
from models.IAED.sIAED import sIAED
from models.IAED.config import config as cIAED
# T2V import
from models.T2V.sT2VRNN import sT2VRNN
from models.T2V.config import CONFIG as cT2V
# CNN import
from models.CNNLSTM.sCNNLSTM import sCNNLSTM
from models.CNNLSTM.config import CONFIG as cCNN
from MyParser import *


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    MODEL = args.model
    MODEL_FOLDER = args.model_dir
    TRAIN_AGENT = args.train_agent
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

    df, features = get_df(TRAIN_AGENT)

    use_att, use_cm, cm, cm_trainable, use_constraint, constraint = cmd_attention_map(args.att, args.catt)


    if MODEL == Models.sIAED.value:
        if TARGETVAR == None: raise ValueError('for models sIAED, target_var needs to be specified')
        # Single-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
        d.downsample(step = 10)
        d.smooth(window_size = 50)
        X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(cIAED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = INITDEC,
                             use_att = use_att, use_cm = use_cm, cm = cm, cm_trainable = cm_trainable, use_constraint = use_constraint, constraint = constraint)
        model = sIAED(config = config)
        model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])


    elif MODEL == Models.sT2V.value:
        if TARGETVAR == None: raise ValueError('for models sT2V, target_var needs to be specified')
        # Single-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
        d.downsample(step = 10)
        d.smooth(window_size = 50)
        X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(cT2V, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = INITDEC,
                             use_att = use_att, use_cm = use_cm, cm = cm, cm_trainable = cm_trainable, use_constraint = use_constraint, constraint = constraint)
        model = sT2VRNN(config = config)
        model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])


    elif MODEL == Models.sCNN.value:
        if TARGETVAR == None: raise ValueError('for models sCNN, target_var needs to be specified')
        # Single-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC, target = TARGETVAR)
        d.downsample(step = 10)
        d.smooth(window_size = 50)
        X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(cCNN, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = INITDEC,
                             use_att = use_att, use_cm = use_cm, cm = cm, cm_trainable = cm_trainable, use_constraint = use_constraint, constraint = constraint)
        model = sCNNLSTM(config = config)
        model.create_model(target_var = TARGETVAR, loss = 'mse', optimizer = Adam(LR), metrics = ['mse', 'mae', 'mape'])


    elif MODEL == Models.mIAED.value:
        # Multi-output data initialization
        d = Data(df, N_PAST, N_DELAY, N_FUTURE, TRAIN_PERC, VAL_PERC, TEST_PERC)
        d.downsample(step = 10)
        d.smooth(window_size = 50)
        X_train, y_train, X_val, y_val, X_test, y_test = d.get_timeseries()

        # IAED Model definition
        config = init_config(cIAED, folder = MODEL_FOLDER, npast = N_PAST, nfuture = N_FUTURE,
                             ndelay = N_DELAY, nfeatures = N_FEATURES, features = features, initDEC = INITDEC,
                             use_att = use_att, use_cm = use_cm, cm = cm, cm_trainable = cm_trainable, use_constraint = use_constraint, constraint = constraint)
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
    model.MAE(X_test, y_test, d.scaler)

    # Model predictions
    model.predict(X_test, y_test, d.scaler, plot = True)