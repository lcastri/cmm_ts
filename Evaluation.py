import numpy as np
from sklearn.metrics import mean_squared_error
from constants import *

from models.utils import get_df
import pickle


if __name__ == "__main__":

    MODELS = ["mIAED_att", "mIAED_PCMCI_f", "mIAED_FPCMCI_f"]
    TEST_AGENTS = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    MRMSE = dict()
    MMAE = dict()
    for M in MODELS:
        MRMSE[M] = {A : float for A in TEST_AGENTS}
        MMAE[M] = {A : float for A in TEST_AGENTS}
    for M in MODELS:
        for a in TEST_AGENTS:
            df, features = get_df(a)
            ya = np.load(RESULT_DIR + '/' + M + '/predictions/ya_npy.npy')
            yp = np.load(RESULT_DIR + '/' + M + '/predictions/yp_npy.npy')
            ae = np.load(RESULT_DIR + '/' + M + '/ae.npy')

            mae = list()
            rmse = list()
            for f in features:
                mae.append(round(ae[:, features.index(f)].mean()/df[f].std(), 3))
                rmse.append(round(mean_squared_error(ya[:,:,features.index(f)], yp[:,:,features.index(f)], squared = False)/df[f].std(), 3))
            MRMSE[M][a]= sum(rmse) / len(rmse)
            MMAE[M][a]= sum(mae) / len(mae)

    with open('RMSE.pkl', 'wb') as file_pi:
        pickle.dump(MRMSE, file_pi)
    with open('MAE.pkl', 'wb') as file_pi:
        pickle.dump(MMAE, file_pi)
    print(MRMSE)
    print(MMAE)
    

    
