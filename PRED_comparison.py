import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from constants import *
from matplotlib import pyplot as plt
from tqdm import tqdm


def create_parser():
    parser = argparse.ArgumentParser(description = 'Multivariate Multistep Timeseries forecasting framework.', formatter_class = RawTextHelpFormatter)
    parser.add_argument("models", nargs = '+', help = "list of model name to compare")
    parser.add_argument("--labels", nargs = '+', help = "list of labels for legend", required = True)
    parser.add_argument("--targetidx", help = "target variable index", required = False, default = "ALL")
    parser.add_argument("--target", help = "target variable name", required = True)
    return parser


def plot_prediction(x, y, yp_dict : dict):
    plt.figure()
    pred_dir = RESULT_DIR + "/PRED_comparison/"
    # Create var folder
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(pred_dir + TARGET):
        os.makedirs(pred_dir + TARGET)

    for t in tqdm(range(len(yp_dict[list(yp_dict.keys())[0]])), desc = TARGETIDX):
        plt.plot(range(t, t + len(x[t][:, TARGETIDX])), x[t][:, TARGETIDX], color = 'peru', label = "past")
        plt.plot(range(t - 1 + len(x[t][:, TARGETIDX]), t - 1 + len(x[t][:, TARGETIDX]) + len(y[t])), y[t], color = 'black', label = "actual")
        for k in yp_dict.keys():
            plt.plot(range(t - 1 + len(x[t][:, TARGETIDX]), t - 1 + len(x[t][:, TARGETIDX]) + len(yp_dict[k][t])), yp_dict[k][t], label = k)
        plt.title("Multi-step prediction - " + TARGET)
        plt.xlabel("step = 0.1s")
        plt.ylabel(TARGET)
        plt.grid()
        plt.legend()
        plt.savefig(pred_dir + "/" + str(TARGET) + "/" + str(t) + ".png")

        plt.clf()
    plt.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    MODELS = args.models
    LABELS = args.labels
    TARGETIDX = int(args.targetidx)
    TARGET = args.target

    plt.figure()
    X = np.load(RESULT_DIR + '/' + MODELS[0] + '/predictions/x_npy.npy')
    Y = np.load(RESULT_DIR + '/' + MODELS[0] + '/predictions/ya_npy.npy')
    Y_pred = dict()
    for m, l in zip(MODELS, LABELS): Y_pred[l] = np.load(RESULT_DIR + '/' + m + '/predictions/yp_npy.npy')
    
    plot_prediction(X, Y, Y_pred)
    
