# Causality-based Multi-step and Multi-output Network for Timeseries Forecasting

Causality based network for multi-step and multi-output timeseries forecasting


## Requirements

* absl==0.0
* absl_py==1.2.0
* keras==2.10.0
* keras_hypetune==0.2.1
* matplotlib==3.6.1
* numpy==1.21.5
* pandas==1.5.0
* scikit_learn==1.1.3
* tensorflow==2.10.0
* tqdm==4.64.1


## Data

CSV file positioned inside folder "data" (to create in main folder).


## Input Specifications

* **model***, [str]: network configuration to use. choices = ['sIAED', 'mIAED'];
* **model_dir***, [str]: model folder that will be created in "training_result" folder;
* **data***, [str]: CSV file to load positioned in "data" folder;
* **npast***, [int]: observation window;
* **nfuture*** [int]: forecasting window;
* **ndelay**, [int]: forecasting delay, default = 0)
* **noinit_dec**, [bool]: use encoder final state as initial state for decoder, default = True;
* **att**, [bool]: use attention mechanism, default = False;
* **catt**, [bool]: use causal-attention [CAUSAL MATRIX, TRAINABLE, CONSTRAINT], default = [None False None];
* **target_var**, [str]: Target variable to forecast (used only if model = sIAED), default = None.<br>
needs to be one of the columns defined in the csv file;
* **percs**, [float float float]: train, val, test percentages, default = [0.7, 0.1, 0.2];
* **patience**, [int]: earlystopping patience, default = 25;
* **batch_size**, [int]: batch size, default = 128;
* **epochs**, [int]: epochs, default = 300;
* **learning_rate**, [float]: learning rate, default = 0.0001;

\* fields are required.

## Example

### Single-output (no attention)

* model_dir = single
* observation window = 32 steps
* forecasting window = 48 steps
* target variable to forecast = d_g
* batch_size = 32

```
python3.8 main.py sIAED single --data data.csv --npast 32 --nfuture 48 --target_var d_g --batch_size 32
```

### Single-output (attention)

* model_dir = single
* observation window = 32 steps
* forecasting window = 48 steps
* target variable to forecast = d_g
* batch_size = 32

```
python3.8 main.py sIAED single --data data.csv --npast 32 --nfuture 48 --target_var d_g --batch_size 32 --att
```

### Single-output (causal attention)

* model_dir = single
* observation window = 32 steps
* forecasting window = 48 steps
* target variable to forecast = d_g
* batch_size = 32

```
python3.8 main.py sIAED single --data data.csv --npast 32 --nfuture 48 --target_var d_g --batch_size 32 --catt CM False None
```

### Multi-output (no attention)

* model_dir = multi
* observation window = 32 steps
* forecasting window = 48 steps
* batch_size = 32

NOTE: in this case target_var is not specified since all the variables in the dataframe are forecasted

```
python3.8 main.py mIAED single --data data.csv --npast 32 --nfuture 48 --batch_size 32
```

### Multi-output (attention)

* model_dir = multi
* observation window = 32 steps
* forecasting window = 48 steps
* batch_size = 32

NOTE: in this case target_var is not specified since all the variables in the dataframe are forecasted

```
python3.8 main.py mIAED single --data data.csv --npast 32 --nfuture 48 --batch_size 32 --att
```

### Multi-output (causal attention)

* model_dir = multi
* observation window = 32 steps
* forecasting window = 48 steps
* batch_size = 32

NOTE: in this case target_var is not specified since all the variables in the dataframe are forecasted

```
python3.8 main.py mIAED single --data data.csv --npast 32 --nfuture 48 --batch_size 32 --catt CM False None
```