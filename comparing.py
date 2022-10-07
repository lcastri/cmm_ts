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
        rmse = np.load(RESULT_DIR + '/' + m + '/rmse.npy')
        if len(rmse) == 1:
            print("RMSE " + l + " : " + str(rmse))
        else:
            plt.plot(rmse, label = l)
    
    plt.xlabel("Time steps")
    plt.ylabel("RMSE")
    if len(rmse) != 1:
        plt.legend()
        plt.grid()
        plt.savefig(RESULT_DIR + '/' + FIGNAME + ".png", dpi = 300)
        plt.savefig(RESULT_DIR + '/' + FIGNAME + ".eps", dpi = 300)
        plt.close()

    
