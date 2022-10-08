import argparse
from argparse import RawTextHelpFormatter
from models.utils import *


def print_init(model, targetvar, modeldir, npast, nfuture, ndelay, initdec, train_perc, val_perc, test_perc,
               use_att, use_cm, cm, cm_trainable, use_constraint, constraint, batch_size, patience, epochs, lr, adjlr):
    
    # PRINT TO CONSOLE
    print("\n#")
    print("# MODEL PARAMETERS")
    print("# model =", model)
    if model == Models.sIAED.value: print("# target var =", targetvar)
    print("# model folder =", modeldir)
    print("# past window steps =", npast)
    print("# future window steps =", nfuture)
    print("# delay steps =", ndelay)
    print("# use encoder state for dec init =", initdec)
    print("# dataset split (train, val, test) =", (train_perc, val_perc, test_perc))
    
    print("#")
    print("# ATTENTION PARAMETERS")
    print("# attention =", use_att)
    if use_att and use_cm and cm_trainable:
        print('# trainable causal model =', cm)
        if use_constraint:
            print("# contraint =", constraint)

    elif use_att and use_cm and not cm_trainable:
        print("# Fixed causal model =", cm)
    
    print("#")
    print("# TRAINING PARAMETERS")
    print("# batch size =", batch_size)
    print("# early stopping patience =", patience)
    print("# epochs =", epochs)
    print("# learning rate =", lr)
    print("# adjust learning rate =", adjlr)
    print("#\n")
    
    # PRINT TO FILE
    f = open(RESULT_DIR + "/" + modeldir + "/parameters.txt", "w")
    f.write("#\n")
    f.write("# MODEL PARAMETERS\n")
    f.write("# model = " + str(model) + "\n")
    if model == Models.sIAED.value: f.write("# target var = " + str(targetvar) + "\n")
    f.write("# model folder = " + str(modeldir) + "\n")
    f.write("# past window steps = " + str(npast) + "\n")
    f.write("# future window steps = " + str(nfuture) + "\n")
    f.write("# delay steps = " + str(ndelay) + "\n")
    f.write("# use encoder state for dec init = " + str(initdec) + "\n")
    f.write("# dataset split (train, val, test) = " + str((train_perc, val_perc, test_perc)) + "\n")
    
    f.write("#" + "\n")
    f.write("# ATTENTION PARAMETERS" + "\n")
    f.write("# attention = " + str(use_att) + "\n")
    if use_att and use_cm and cm_trainable:
        f.write("# trainable causal model = " + cm + "\n")
        if use_constraint:
            f.write("# contraint = " + str(constraint) + "\n")

    elif use_att and use_cm and not cm_trainable:
        f.write("# Fixed causality = " + cm + "\n")
    
    f.write("#" + "\n")
    f.write("# TRAINING PARAMETERS" + "\n")
    f.write("# batch size = " + str(batch_size) + "\n")
    f.write("# early stopping patience = " + str(patience) + "\n")
    f.write("# epochs = " + str(epochs) + "\n")
    f.write("# learning rate = " + str(lr) + "\n")
    f.write("# adjust learning rate = " + str(adjlr) + "\n")
    f.write("#\n")
    f.close()


def create_parser():
    model_description = "\n".join([k + " - " + MODELS[k] for k in MODELS])

    parser = argparse.ArgumentParser(description = 'Multivariate Multistep Timeseries forecasting framework.', formatter_class = RawTextHelpFormatter)
    parser.add_argument("model", type = str, choices = list(MODELS.keys()), help = model_description)
    parser.add_argument("model_dir", type = str, help = "model folder")
    parser.add_argument("--npast", type = int, help = "observation window", required = True)
    parser.add_argument("--nfuture", type = int, help = "forecasting window", required = True)
    parser.add_argument("--ndelay", type = int, help = "forecasting delay [default 0]", required = False, default = 0)
    parser.add_argument("--noinit_dec", action = 'store_false', help = "use ENC final state as init for DEC bit [default True]", required = False, default = True)
    parser.add_argument("--att", action = 'store_true', help = "use attention bit [default False]", required = False, default = False)
    parser.add_argument("--catt", nargs = 3, help = "use causal-attention [CAUSAL MATRIX, TRAINABLE, CONSTRAINT] [default None False None]", required = False, default = [None, False, None])
    parser.add_argument("--target_var", type = str, help = "Target variable to forecast [used only if model = sIAED/sT2V] [default None]", required = False, default = None)
    parser.add_argument("--percs", nargs = 3, action='append', help = "[train, val, test[] percentages [default [0.6, 0.2, 0.2]]", required = False, default = [0.6, 0.2, 0.2])
    parser.add_argument("--patience", type = int, help = "earlystopping patience [default 10]", required = False, default = 25)
    parser.add_argument("--batch_size", type = int, help = "batch size [default 128]", required = False, default = 128)
    parser.add_argument("--epochs", type = int, help = "epochs [default 300]", required = False, default = 300)
    parser.add_argument("--learning_rate", type = float, help = "learning rate [default 0.0001]", required = False, default = 0.0001)
    parser.add_argument("--adjLR", nargs = 3, help = "Modifying learning rate strategy (freq[epochs], factor, justOnce)", required = False)
    return parser