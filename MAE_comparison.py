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
    return parser


if __name__ == "__main__":
    debug = False
    if debug:
        MODELS = ["PROVA12", "PROVA13"]
        LABELS = ["prova12", "prova13"]
        FIGNAME = "ciao"
        TRAIN_AGENT = 11
    else:
        parser = create_parser()
        args = parser.parse_args()
        MODELS = args.models
        LABELS = args.labels
        FIGNAME = args.figname
        TRAIN_AGENT = args.train_agent
    df, features = get_df(TRAIN_AGENT)

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

    
