import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


ALL = 'all'


class Data():
    def __init__(self,
                 data: pd.DataFrame,
                 n_past: int, 
                 n_delay: int, 
                 n_future: int,
                 train_prec: float,
                 val_prec: float,
                 test_prec: float,
                 target: str = ALL):
        
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
        
        self.target = target
        self.scaler = None
        # self.scalerIN = None
        # self.scalerOUT = None


    def get_sets(self, seq):
        train_len = int(len(self.data) * self.train_perc)
        val_len = int(len(self.data) * self.val_perc)
        test_len = int(len(self.data) * self.test_perc)
        return seq[:train_len], seq[train_len:train_len + val_len], seq[train_len + val_len:]


    def downsample(self, step):
        self.data = pd.DataFrame(self.data.values[::step, :], columns=self.data.columns)


    def scale_data(self):
        df = self.data.astype(float)
        self.scaler = MinMaxScaler()
        self.scaler = self.scaler.fit(df)
        self.data_scaled = self.scaler.transform(df)
        # df = self.data.astype(float)
        # self.scaler = MinMaxScaler()
        # self.scaler = self.scaler.fit(df)
        # self.input_scaled = self.scaler.transform(df)

        # if self.target != ALL:
        #     out_seq = np.array(df[self.target]).reshape((len(df[self.target]), -1))
        # self.scalerOUT = MinMaxScaler()
        # self.scalerOUT = self.scalerOUT.fit(out_seq if self.target != ALL else df)
        # self.output_scaled = self.scalerOUT.transform(out_seq if self.target != ALL else df)


    def split_sequence(self):
        X, y = list(), list()
        for i in range(len(self.data_scaled)): 
            lag_end = i + self.n_past
            forecast_end = self.n_delay + lag_end + self.n_future
            if forecast_end > len(self.data_scaled):
                break
            if self.target == ALL:
                seq_x, seq_y = self.data_scaled[i:lag_end], self.data_scaled[self.n_delay + lag_end:forecast_end]
            else:
                t = list(self.data.columns).index(self.target)
                seq_x, seq_y = self.data_scaled[i:lag_end], self.data_scaled[self.n_delay + lag_end:forecast_end, t]
            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)
        if self.target != ALL:
            y = y.reshape((y.shape[0], y.shape[1], -1))
        return X, y


    def get_timeseries(self):
        self.scale_data()
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
        


# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# class Data():
#     def __init__(self,
#                  data: pd.DataFrame,
#                  n_past: int, 
#                  n_delay: int, 
#                  n_future: int,
#                  train_prec: float,
#                  val_prec: float,
#                  test_prec: float):
        
#         # Data
#         self.data = data
#         self.data_scaled = None
#         self.features = data.columns
#         self.N = len(self.features)

#         # Data parameters
#         self.n_past = n_past         
#         self.n_delay = n_delay      
#         self.n_future = n_future 

#         # Splittng percentages
#         self.train_perc = train_prec
#         self.val_perc = val_prec
#         self.test_perc = test_prec

#         self.scaler = None


#     def get_sets(self):
#         train_len = int(len(self.data_scaled) * self.train_perc)
#         val_len = int(len(self.data_scaled) * self.val_perc)
#         test_len = int(len(self.data_scaled) * self.test_perc)
#         return self.data_scaled[:train_len], self.data_scaled[train_len:train_len + val_len], self.data_scaled[train_len + val_len:]

#     def downsample(self, step):
#         self.data = pd.DataFrame(self.data.values[::step, :], columns=self.data.columns)

#     def scale_data(self):
#         df = self.data.astype(float)
#         self.scaler = MinMaxScaler()
#         self.scaler = self.scaler.fit(df)
#         self.data_scaled = self.scaler.transform(df)


#     def split_sequence(self, sequence):
#         X, y = list(), list()
#         for i in range(len(sequence)): 
#             lag_end = i + self.n_past
#             forecast_end = self.n_delay + lag_end + self.n_future
#             if forecast_end > len(sequence):
#                 break
#             seq_x, seq_y = sequence[i:lag_end], sequence[self.n_delay + lag_end:forecast_end]
#             X.append(seq_x)
#             y.append(seq_y)
#         return np.array(X), np.array(y)


#     def get_timeseries(self):
#         train, val, test = self.get_sets()
#         X_train, y_train = self.split_sequence(train)
#         X_val, y_val = self.split_sequence(val)
#         X_test, y_test = self.split_sequence(test)
#         print("X train shape", X_train.shape)
#         print("y train shape", y_train.shape)
#         print("X val shape", X_val.shape)
#         print("y val shape", y_val.shape)
#         print("X test shape", X_test.shape)
#         print("y test shape", y_test.shape)

#         return X_train, y_train, X_val, y_val, X_test, y_test