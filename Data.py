import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Data():
    def __init__(self,
                 data: pd.DataFrame,
                 n_past: int, 
                 n_delay: int, 
                 n_future: int,
                 train_prec: float,
                 val_prec: float,
                 test_prec: float):
        
        # Data
        self.data = data
        self.data_scaled = None
        self.features = data.columns
        self.N = len(self.features)

        # Data parameters
        self.n_past = n_past         
        self.n_delay = n_delay      
        self.n_future = n_future 

        # Splittng percentages
        self.train_perc = train_prec
        self.val_perc = val_prec
        self.test_perc = test_prec

        self.scaler = None


    def get_sets(self):
        train_len = int(len(self.data_scaled) * self.train_perc)
        val_len = int(len(self.data_scaled) * self.val_perc)
        test_len = int(len(self.data_scaled) * self.test_perc)
        return self.data_scaled[:train_len], self.data_scaled[train_len:train_len + val_len], self.data_scaled[train_len + val_len:]


    def scale_data(self):
        df = self.data.astype(float)
        self.scaler = MinMaxScaler()
        self.scaler = self.scaler.fit(df)
        self.data_scaled = self.scaler.transform(df)


    # def get_Xy(self, sequences):
    #     X, y = list(), list()
    #     for i in range(len(sequences)):
    #         # find the end of this pattern
    #         end_ix = i + self.n_past
    #         out_end_ix = self.n_delay + end_ix + self.n_future
    #         # check if we are beyond the dataset
    #         if out_end_ix > len(sequences):
    #             break
    #         # gather input and output parts of the pattern
    #         seq_x = sequences[i : end_ix, :]
    #         seq_y = sequences[self.n_delay + end_ix - 1 : out_end_ix, :]

    #         X.append(seq_x)
    #         y.append(seq_y)
    #     return np.array(X), np.squeeze(np.array(y))


    # def get_timeseries(self):
    #     train, val, test = self.get_sets()
    #     trainX_inputs = list()
    #     trainY_inputs = list()
    #     valX_inputs = list()
    #     valY_inputs = list()
    #     testX_inputs = list()
    #     testY_inputs = list()
    #     for var in range(self.N):
    #         trainX, trainY = self.get_Xy(train, var)
    #         valX, valY = self.get_Xy(val, var)
    #         testX, testY = self.get_Xy(test, var)
    #         trainX_inputs.append(trainX)
    #         trainY_inputs.append(trainY)
    #         valX_inputs.append(valX)
    #         valY_inputs.append(valY)
    #         testX_inputs.append(testX)
    #         testY_inputs.append(testY)
    #     return trainX_inputs, trainY_inputs, valX_inputs, valY_inputs, testX_inputs, testY_inputs

    def split_sequence(self, sequence):
        X, y = list(), list()
        for i in range(len(sequence)): 
            lag_end = i + self.n_past
            forecast_end = self.n_delay + lag_end + self.n_future
            if forecast_end > len(sequence):
                break
            seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.squeeze(np.array(y))


    def get_timeseries(self):
        train, val, test = self.get_sets()
        X_train, y_train = self.split_sequence(train)
        X_val, y_val = self.split_sequence(val)
        X_test, y_test = self.split_sequence(test)
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)
        print(X_test.shape)
        print(y_test.shape)

        return X_train, y_train, X_val, y_val, X_test, y_test
        
