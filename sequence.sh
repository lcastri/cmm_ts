#!/bin/sh
python3.8 main.py sIAED sIAED_P20_F150_catt_f_1000 --npast 20 --nfuture 150 --target_var d_g --adjLR 150 0.1 True --catt_f --epochs 1000
python3.8 main.py sIAED sIAED_P20_F150_catt_t_1000 --npast 20 --nfuture 150 --target_var d_g --adjLR 150 0.1 True --catt_t --epochs 1000
python3.8 main.py sIAED sIAED_P20_F150_att_1000 --npast 20 --nfuture 150 --target_var d_g --adjLR 150 0.1 True --att --epochs 1000
python3.8 main.py sIAED sIAED_P20_F150_1000 --npast 20 --nfuture 150 --target_var d_g --adjLR 150 0.1 True --epochs 1000
