import pandas as pd

# load csv and remove NaNs
csv_path = "data/Exp_1_run_1/agent_11.csv"
df = pd.read_csv(csv_path)
df.fillna(method="ffill", inplace = True)
df.fillna(method="bfill", inplace = True)

# Parameters definition
N_FUTURE = 1
N_PAST = 320
# N_DELAY = 480
N_DELAY = 0
N_FEATURES = 8
TRAIN_PERC = 0.6
VAL_PERC = 0.25
TEST_PERC = 0.15
MODEL_NAME = "model_CALSTM_nodelay_causal_fpcmci"

