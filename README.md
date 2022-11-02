# Causality-based Multi-step and Multi-output Network for Timeseries Forecasting

Causality based network for multi-step and multi-output timeseries forecasting<br>

<p float="left">
    <img src="https://github.com/lcastri/timeseries_forecasting/blob/main/images/d_g.gif" width="24.2%" height="24.2%" />
    <img src="https://github.com/lcastri/timeseries_forecasting/blob/main/images/v.gif" width="24.2%" height="24.2%" />
    <img src="https://github.com/lcastri/timeseries_forecasting/blob/main/images/risk.gif" width="24.2%" height="24.2%" />
    <img src="https://github.com/lcastri/timeseries_forecasting/blob/main/images/theta_g.gif" width="24.2%" height="24.2%" />
    <br>
    <img src="https://github.com/lcastri/timeseries_forecasting/blob/main/images/omega.gif" width="24.2%" height="24.2%" />
    <img src="https://github.com/lcastri/timeseries_forecasting/blob/main/images/theta.gif" width="24.2%" height="24.2%" />
    <img src="https://github.com/lcastri/timeseries_forecasting/blob/main/images/g_seq.gif" width="24.2%" height="24.2%" />
    <img src="https://github.com/lcastri/timeseries_forecasting/blob/main/images/d_obs.gif" width="24.2%" height="24.2%" />
</p>


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

|               |           type          | required |      default      | description                                                                                                         |
|---------------|:-----------------------:|:--------:|:-----------------:|---------------------------------------------------------------------------------------------------------------------|
| model         |           str           |   True   |         -         | network configuration to use. choices = ['sIAED', 'mIAED']                                                          |
| model_dir     |           str           |   True   |         -         | model folder that will be created in "training_result" folder                                                       |
| data          |           str           |   True   |         -         | CSV file to load positioned in "data" folder                                                                        |
| npast         |           int           |   True   |         -         | observation window                                                                                                  |
| nfuture       |           int           |   True   |         -         | forecasting window                                                                                                  |
| ndelay        |           int           |   False  |         0         | forecasting delay                                                                                                   |
| noinit_dec    |           bool          |   False  |        True       | use encoder final state as initial state for decoder                                                                |
| att           |           bool          |   False  |       False       | use attention mechanism                                                                                             |
| catt          | [str, bool, float] |   False  | [None False None] | use causal-attention mechanism. Explained in detail [here](#causal-attention-mechanism).                                                                                       |
| target_var    |           str           |   False  |        None       | Target variable to forecast (used only if model = sIAED). Needs to match one of the columns defined in the csv file |
| percs         |  [float, float, float]  |   False  |   [0.7 0.1 0.2]   | TRAIN, VAL, TEST percentage                                                              |
| patience      |           int           |   False  |         25        | earlystopping patience                                                                                              |
| batch_size    |           int           |   False  |        128        | batch size                                                                                                          |
| epochs        |           int           |   False  |        300        | epochs                                                                                                              |
| learning_rate |          float          |   False  |       0.0001      | learning rate                                                                                                       |

## Causal Attention mechanism

Before using the **--catt** option, we need to define the causal matrix extracted from our causal model in [constants.py](https://github.com/lcastri/timeseries_forecasting/blob/main/constants.py) as follows:

considering a system of 4 variables
```
causal_matrix = np.array([0.79469, 0.07976, 0, 0.20714],
                         [0, 0.54711, 0.11897, 0],
                         [0, 0, 0.99110, 0],
                         [0.06849, 0.06341, 0, 0.97200])
```
and then, we need to create a link from a string (our input in --catt) to the causal matrix:
```
class CausalModel(Enum):
    CM = "CM"

CAUSAL_MODELS = {CausalModel.CM.value : causal_matrix}
```
Now we are ready to use the --catt option. It needs to be followed by three inputs: <CAUSAL_MATRIX> <TRAINABLE_FLAG> <TRAIN_CONSTRAINT>. In particular:<br>
* <CAUSAL_MATRIX> : string linked to the causal_matrix defined in constants.py script. In this case "CM";
* <TRAINABLE_FLAG> : flag to set the causal matrix as a trainable network parameter. If False, the causal matrix will not be trained;
* <TRAIN_CONSTRAINT> : training threshold for each value componing the causal matrix. When using the causal matrix as trainable parameter, it helps to maintain the values of the post-training causal matrix close to the pre-training one. The constraint is defined as follows:

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\Large&space;\color{White}{x&space;-&space;T&space;\leq&space;x&space;\leq&space;x&plus;T}" /> 
</p>

where $x$ is a value of the causal matrix and $T$ the <TRAIN_CONSTRAINT>. The constraints let the network to change the causal value by a certain quantity, specified by T, but not to diverge. If <TRAINABLE_FLAG> is False, this field does not have effect.

#### Non-trainable causal matrix 

```
--catt CM False None
```

#### Trainable causal matrix (without constraint)

```
--catt CM True None
```

#### Trainable causal matrix (with constraint)

```
--catt CM True 0.1
```

## Examples

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
