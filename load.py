
from models.sIAED.sIAED import sIAED


m = sIAED(folder = "sIAED_P20_F150_prova")
m.RMSE()

# model.save_cmatrix()

# # Model evaluation
# model.RMSE(x_test, y_test, d.scaler)

# # Model predictions
# model.plot_predictions(x_test, y_test, d.scaler)