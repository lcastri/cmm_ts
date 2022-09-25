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

        self.scalerIN = None
        self.scalerOUT = None


    def get_sets(self, seq):
        train_len = int(len(self.data) * self.train_perc)
        val_len = int(len(self.data) * self.val_perc)
        test_len = int(len(self.data) * self.test_perc)
        return seq[:train_len], seq[train_len:train_len + val_len], seq[train_len + val_len:]

    def downsample(self, step):
        self.data = pd.DataFrame(self.data.values[::step, :], columns=self.data.columns)

    def scale_data(self, target):
        df = self.data.astype(float)
        self.scalerIN = MinMaxScaler()
        self.scalerIN = self.scalerIN.fit(df)
        self.input_scaled = self.scalerIN.transform(df)

        out_seq = np.array(df[target]).reshape((len(df[target]), -1))
        self.scalerOUT = MinMaxScaler()
        self.scalerOUT = self.scalerOUT.fit(out_seq)
        self.output_scaled = self.scalerOUT.transform(out_seq)

    # def scale_input(self, seq):
    #     seq = seq.astype(float)
    #     self.scalerIN = MinMaxScaler()
    #     self.scalerIN = self.scalerIN.fit(seq)
    #     return self.scalerIN.transform(seq)

    # def scale_output(self, seq):
    #     seq = seq.astype(float)
    #     self.scalerOUT = MinMaxScaler()
    #     self.scalerOUT = self.scalerOUT.fit(seq)
    #     return self.scalerOUT.transform(seq)

    def split_sequence(self):
        X, y = list(), list()
        for i in range(len(self.input_scaled)): 
            lag_end = i + self.n_past
            forecast_end = self.n_delay + lag_end + self.n_future
            if forecast_end > len(self.input_scaled):
                break
            seq_x, seq_y = self.input_scaled[i:lag_end], self.output_scaled[self.n_delay + lag_end:forecast_end]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


    def get_timeseries(self, target):
        self.scale_data(target)
        X, y = self.split_sequence()
        X_train, X_val, X_test = self.get_sets(X)
        y_train, y_val, y_test = self.get_sets(y)
        print("X train shape", X_train.shape)
        print("y train shape", y_train.shape)
        print("X val shape", X_val.shape)
        print("y val shape", y_val.shape)
        print("X test shape", X_test.shape)
        print("y test shape", y_test.shape)

        return X_train, y_train, X_val, y_val, X_test, y_test
        
