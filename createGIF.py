import glob
from PIL import Image
from natsort import natsorted


LIST_FEATURES = ['d_g', 'v', 'risk', 'theta_g', 'omega', 'theta', 'g_seq', 'd_obs']
# LIST_FEATURES = ['d_g']
for f in LIST_FEATURES:
    print(f)
    # filepaths
    fp_in = "/home/lcastri/Git/timeseries_forecasting/training_result/256_mIAED_FPCMCI_f/predictions/" + f + "/*.png"
    fp_out = "/home/lcastri/Git/timeseries_forecasting/training_result/256_mIAED_FPCMCI_f/" + f + ".gif"

    imgs = (Image.open(f) for f in natsorted(glob.glob(fp_in)))
    img = next(imgs)  # extract first image from iterator
    img.save(fp = fp_out, format = 'GIF', append_images = imgs,
             save_all = True, duration = 50, loop = 1)