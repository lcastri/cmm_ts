from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') 
from matplotlib import pyplot as plt
# import tkinter

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


    def get_sets(self, seq):
        train_len = int(len(self.data) * self.train_perc)
        val_len = int(len(self.data) * self.val_perc)
        test_len = int(len(self.data) * self.test_perc)
        return seq[:train_len], seq[train_len:train_len + val_len], seq[train_len + val_len:]


    def downsample(self, step):
        self.data = pd.DataFrame(self.data.values[::step, :], columns=self.data.columns)


    def augment(self, nrepeat = 5, sigma = 0.05, scaling = 0.5):
        """
        data augmentation adding gaussian noise and scaling data

        Args:
            nrepeat (int, optional): Number of concatenation of the same dataset. Defaults to 5.
            sigma (float, optional): Gaussian noise sigma to apply to each repetition. Defaults to 0.1.
        """
        np.random.seed(0)

        list_d = list()
        for _ in range(nrepeat):
            d = deepcopy(self.data)
            noise = np.random.normal(0, sigma, size = self.data.shape)
            scaling_factor = np.random.uniform(scaling, 1)
            rep = scaling_factor * d + noise
            rep['g_seq'] = self.data['g_seq']
            list_d.append(rep)
        self.data = pd.concat(list_d, ignore_index = True)

    
    def plot_ts(self):
        self.data.plot(subplots=True)

        plt.tight_layout()
        plt.show()

    
    def scale_data(self):
        df = self.data.astype(float)
        self.scaler = MinMaxScaler()
        self.scaler = self.scaler.fit(df)
        self.data_scaled = self.scaler.transform(df)


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
    