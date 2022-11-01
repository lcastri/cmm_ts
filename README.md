# Causality-based Multi-step and Multi-output Network for Timeseries Forecasting

Causality based network for multi-step and multi-output timeseries forecasting<br><br>

<p float="left">
    <a href="https://drive.google.com/uc?export=view&id=1W3KsS34zMLn2eRkvBZNvy_KwHIa9lZBP"><img src="https://drive.google.com/uc?export=view&id=1W3KsS34zMLn2eRkvBZNvy_KwHIa9lZBP" style="width: 650px; width: 24.2%; height: auto" />
    <a href="https://drive.google.com/uc?export=view&id=1Airea-32wIkR98Ylvbh9OeF_-lfm2f_e"><img src="https://drive.google.com/uc?export=view&id=1Airea-32wIkR98Ylvbh9OeF_-lfm2f_e" style="width: 650px; width: 24.2%; height: auto" />
    <a href="https://drive.google.com/uc?export=view&id=19fwB4f3ZLkGtjpjKKN8wphW7sJyeYe_h"><img src="https://drive.google.com/uc?export=view&id=19fwB4f3ZLkGtjpjKKN8wphW7sJyeYe_h" style="width: 650px; width: 24.2%; height: auto" />
    <a href="https://drive.google.com/uc?export=view&id=1uUOn9tpVzN0HeZOdtdw6KlFRvqv9nXE8"><img src="https://drive.google.com/uc?export=view&id=1uUOn9tpVzN0HeZOdtdw6KlFRvqv9nXE8" style="width: 650px; width: 24.2%; height: auto" />
    <br>
    <a href="https://drive.google.com/uc?export=view&id=1prSMeKqgEvOkVij2S8eA8b16LeMbdlBV"><img src="https://drive.google.com/uc?export=view&id=1prSMeKqgEvOkVij2S8eA8b16LeMbdlBV" style="width: 650px; width: 24.2%; height: auto" />
    <a href="https://drive.google.com/uc?export=view&id=1qsgUep12pNIZ_ChelStLigzxa9UQJ8hK"><img src="https://drive.google.com/uc?export=view&id=1qsgUep12pNIZ_ChelStLigzxa9UQJ8hK" style="width: 650px; width: 24.2%; height: auto" />
    <a href="https://drive.google.com/uc?export=view&id=1zsXgfwv289cQ7C0zsdn82YCoeb0Wn-Fr"><img src="https://drive.google.com/uc?export=view&id=1zsXgfwv289cQ7C0zsdn82YCoeb0Wn-Fr" style="width: 650px; width: 24.2%; height: auto" />
    <a href="https://drive.google.com/uc?export=view&id=1Or3bCWuevLQzZ_B__LEu4Y2dJ8oE9LAh"><img src="https://drive.google.com/uc?export=view&id=1Or3bCWuevLQzZ_B__LEu4Y2dJ8oE9LAh" style="width: 650px; width: 24.2%; height: auto" />
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
