import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from constants import *
from matplotlib import pyplot as plt


def create_parser():
    parser = argparse.ArgumentParser(description = 'Multivariate Multistep Timeseries forecasting framework.', formatter_class = RawTextHelpFormatter)
    parser.add_argument("models", nargs = '+', help = "list of model name to compare")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    MODELS = args.models

    plt.figure()
    for m in MODELS:
        rmse = np.load(RESULT_DIR + '/' + m + '/rmse.npy')
        plt.plot(rmse, label = m)
        
    plt.legend()
    plt.grid()
    fig_name = '__'.join(MODELS)
    plt.savefig(RESULT_DIR + '/' + fig_name + "__rmse_compare.png", dpi = 300)
    plt.savefig(RESULT_DIR + '/' + fig_name + "__rmse_compare.eps", dpi = 300)
    plt.close()

    
