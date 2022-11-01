import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from constants import *
from matplotlib import pyplot as plt

from models.utils import get_df


def create_parser():
    parser = argparse.ArgumentParser(description = 'Multivariate Multistep Timeseries forecasting framework.', formatter_class = RawTextHelpFormatter)
    parser.add_argument("models", nargs = '+', help = "list of model name to compare")
    parser.add_argument("--labels", nargs = '+', help = "list of labels for legend", required = True)
    parser.add_argument("--figname", help = "figure name", required = True)
    parser.add_argument("--train_agent", type = int, choices = list(range(3, 12)), help = "agent training data", required = True)
    parser.add_argument("--target_var", type = str, help = "Target variable to forecast [used only if model = sIAED/sT2V/sCNN] [default None]", required = False, default = None)

    return parser




    
if __name__ == "__main__":
    # MODELS = ["sIAED", "sIAED_att", "sIAED_PCMCI_t", "sIAED_FPCMCI_t"]
    # MODELS = ["sIAED", "sIAED_att", "sIAED_PCMCI_f", "sIAED_FPCMCI_f"]
    MODELS = ["sIAED", "sIAED_att", "sIAED_PCMCI_f", "sIAED_PCMCI_t", "sIAED_FPCMCI_f", "sIAED_FPCMCI_t"]
    # LABELS = ["noatt", "att", "PCMCI_t", "FPCMCI_t"]
    # LABELS = ["noatt", "att", "PCMCI_f", "FPCMCI_f"]
    LABELS = ["noatt", "att", "PCMCI_f", "PCMCI_t", "FPCMCI_f", "FPCMCI_t"]
    # FIGNAME = "sIAED_t_128"
    # FIGNAME = "sIAED_f_128"
    FIGNAME = "sIAED_128"
    TRAIN_AGENT = 11
    TARGETVAR = 'd_g'
    df, features = get_df(TRAIN_AGENT)

    if TARGETVAR is not None:
        plt.figure()
        for m, l in zip(MODELS, LABELS):
            ae = np.load(RESULT_DIR + '/' + m + '/ae.npy')
            plt.plot(ae[:, features.index(TARGETVAR)], label = l + " NMAE: " + str(round(ae[:, features.index(TARGETVAR)].mean()/df[TARGETVAR].std(), 3)))
            
        plt.xlabel("Time steps [0.1s]")
        plt.ylabel("Abs error")
        plt.title(TARGETVAR + " abs error")
        plt.legend()
        plt.grid()
        plt.savefig(RESULT_DIR + '/' + FIGNAME + '_' + TARGETVAR + ".png", dpi = 300)
        plt.savefig(RESULT_DIR + '/' + FIGNAME + '_' + TARGETVAR + ".eps", dpi = 300)
        plt.close()

    else:
        for f in features:
            plt.figure()
            for m, l in zip(MODELS, LABELS):
                ae = np.load(RESULT_DIR + '/' + m + '/ae.npy')
                plt.plot(ae[:, features.index(f)], label = l + " NMAE: " + str(round(ae[:, features.index(f)].mean()/df[f].std(), 3)))
            
            plt.xlabel("Time steps [0.1s]")
            plt.ylabel("Abs error")
            plt.title(f + " abs error")
            plt.legend()
            plt.grid()
            plt.savefig(RESULT_DIR + '/' + FIGNAME + '_' + f + ".png", dpi = 300)
            plt.savefig(RESULT_DIR + '/' + FIGNAME + '_' + f + ".eps", dpi = 300)
            plt.close()