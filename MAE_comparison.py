import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from constants import *
from matplotlib import pyplot as plt


def create_parser():
    parser = argparse.ArgumentParser(description = 'Multivariate Multistep Timeseries forecasting framework.', formatter_class = RawTextHelpFormatter)
    parser.add_argument("models", nargs = '+', help = "list of model name to compare")
    parser.add_argument("--labels", nargs = '+', help = "list of labels for legend", required = True)
    parser.add_argument("--figname", help = "figure name", required = True)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    MODELS = args.models
    LABELS = args.labels
    FIGNAME = args.figname

    plt.figure()
    for m, l in zip(MODELS, LABELS):
        mae = np.load(RESULT_DIR + '/' + m + '/mae.npy')
        mae_mean = np.mean(mae)
        if len(mae) == 1:
            print("MAE " + l + " : " + str(mae))
        else:
            plt.plot(mae, label = l + " MAE: " + str(round(mae_mean, 3)))
    
    plt.xlabel("Time steps [0.1s]")
    plt.ylabel("Abs error")
    if len(mae) != 1:
        plt.legend()
        plt.grid()
        plt.savefig(RESULT_DIR + '/' + FIGNAME + ".png", dpi = 300)
        plt.savefig(RESULT_DIR + '/' + FIGNAME + ".eps", dpi = 300)
        plt.close()

    
