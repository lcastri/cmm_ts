
from models.sIAED import sIAED
import numpy as np

m = sIAED(folder = "PCMCI_encdec256_catt_t_initDEC") #FPCMCI_encdec256_catt_t_initDEC PCMCI_encdec256_catt_t_initDEC
m.mean_RMSE(np.load(m.model_dir +  "/rmse.npy"))

# model.save_cmatrix()

# # Model evaluation
# model.RMSE(x_test, y_test, d.scaler)

# # Model predictions
# model.plot_predictions(x_test, y_test, d.scaler)