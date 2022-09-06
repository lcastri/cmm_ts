from keras.models import *
from keras.layers import *
from parameters import *
from attention import CausalAttention
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

        
class CA_LSTM():
    def __init__(self,
                 features: list,
                 units: int,
                 n_past: int, 
                 use_attention_layer: bool = True, 
                 causal_matrix = None):


        if use_attention_layer and causal_matrix is None: 
            raise ValueError('if use_attention_layer == True then causal_matrix cannot be None')
              
        self.features = features
        self.n_past = n_past         
        self.n_features = len(features)
        self.units = units

        input_layer = Input(shape=(self.n_past, self.n_features))

        # adding n_feature parallel input - causal_attention - LSTM blocks
        cas = list()
        lstms = list()
        for var in self.features:           
            if use_attention_layer:
                cas.append(CausalAttention(causal_matrix[features.index(var), :], name = str(var) + '_causal_attention')(input_layer))
            lstms.append(LSTM(self.units, 
                              name = str(var) + '_lstm', 
                              activation='tanh', 
                              return_sequences = False, 
                              input_shape = (self.n_past, self.n_features))(cas[self.features.index(var)] if use_attention_layer else input_layer))

        concat = concatenate(lstms) 
        dropout = Dropout(0.5)(concat)
        # output = TimeDistributed(Dense(self.n_features, activation='linear', name='dense')(dropout))
        output = Dense(self.n_features, activation='linear', name='dense')(dropout)
        self.model = Model(inputs = input_layer, outputs = output)
        self.model.compile(loss='mse', optimizer = Adam(0.00005), metrics=['mse', 'mae', 'mape'])
        self.model.summary()
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


    def fit(self, n_epochs, train, val, callbacks = list()):
        trainX, trainY = train
        valX, valY = val
        history = self.model.fit(x = trainX, y = trainY, validation_data = (valX, valY), batch_size = 128,
                                 epochs = n_epochs, callbacks = callbacks)
        plt.figure()
        plt.plot(history.history["loss"], label = "Training loss")
        plt.plot(history.history["val_loss"], label = "Validation loss")
        plt.legend()

        plt.figure()
        plt.plot(history.history["mae"], label = "Training mae")
        plt.plot(history.history["val_mae"], label = "Validation mae")
        plt.legend()

        plt.figure()
        plt.plot(history.history["mape"], label = "Training mape")
        plt.plot(history.history["val_mape"], label = "Validation mape")
        plt.legend()

        plt.show()