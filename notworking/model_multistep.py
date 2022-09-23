from keras.models import *
from keras.layers import *
from parameters import *
from notworking.attention import CausalAttention
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import os

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
class CA_LSTM(Model):
    def __init__(self,
                 folder: str,
                 features: list,
                 list_units: list,
                 n_past: int, 
                 n_future: int, 
                 use_attention_layer: bool = True, 
                 causal_matrix = None,
                 causal_config = None):

        super().__init__()
        if use_attention_layer and causal_matrix is None: 
            raise ValueError('if use_attention_layer == True then causal_matrix cannot be None')
        
        self.folder = folder
        self.features = features
        self.n_past = n_past         
        self.n_future = n_future       
        self.n_features = len(features)
        self.units = list_units

        create_folder(self.folder)

        input_layer = Input(shape=(self.n_past, self.n_features))

        # adding n_feature parallel causal_attention - LSTM blocks
        head_list = []
        for var in self.features:           
            if use_attention_layer:
                ca = CausalAttention(causal_matrix[features.index(var), :], name = str(var) + '_causal_attention', config = causal_config)(input_layer)
            lstm_enc, h, c = LSTM(self.units[0], name = str(var) + '_lstm_encoder', activation='tanh', 
                           return_sequences = False, return_state=True, input_shape = (self.n_past, self.n_features))(ca if use_attention_layer else input_layer)
            repeat = RepeatVector(self.n_future, name = str(var) + '_repeat')(lstm_enc)
            lstm_dec = LSTM(self.units[1], name = str(var) + '_lstm_decoder', activation='tanh', 
                           return_sequences=True)(repeat)
            head_list.append(lstm_dec)

        concat = concatenate(head_list) 
        dropout = Dropout(0.2)(concat)
        output = TimeDistributed(Dense(self.n_features, activation='linear', name='dense'))(dropout)
        self.model = Model(inputs = input_layer, outputs = output)
        self.model.compile(loss='mse', optimizer = Adam(0.00005), metrics=['mse', 'mae', 'mape'], run_eagerly = True)
        self.model.summary()
        plot_model(self.model, to_file=self.folder+'model_plot.png', show_shapes=True, show_layer_names=True)


    def fit(self, n_epochs, train, val, callbacks = list()):
        trainX, trainY = train
        valX, valY = val
        history = self.model.fit(x = trainX, y = trainY, validation_data = (valX, valY), batch_size = 128,
                                 epochs = n_epochs, callbacks = callbacks)
        plt.figure()
        plt.plot(history.history["loss"], label = "Training loss")
        plt.plot(history.history["val_loss"], label = "Validation loss")
        plt.legend()
        plt.savefig(self.folder + "loss.png", dpi = 300)
        plt.savefig(self.folder + "loss.eps", dpi = 300)

        plt.figure()
        plt.plot(history.history["mae"], label = "Training mae")
        plt.plot(history.history["val_mae"], label = "Validation mae")
        plt.legend()
        plt.savefig(self.folder + "mae.png", dpi = 300)
        plt.savefig(self.folder + "mae.eps", dpi = 300)

        plt.figure()
        plt.plot(history.history["mape"], label = "Training mape")
        plt.plot(history.history["val_mape"], label = "Validation mape")
        plt.legend()
        plt.savefig(self.folder + "mape.png", dpi = 300)
        plt.savefig(self.folder + "mape.eps", dpi = 300)


    def get_config(self):
        return {"model": self.model}


    @classmethod
    def from_config(cls, config):
        return cls(**config)