from keras.models import *
from keras.layers import *
from parameters import *
from Attention import CAttention
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import os
import tensorflow as tf


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
class CA_LSTM(Model):
    def __init__(self,
                 folder: str,
                 features: list,
                 units: list,
                 n_past: int, 
                 n_future: int, 
                 batch_size: int, 
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
        self.batch_size = batch_size
        self.units = units

        self.past_states = dict()
        self.new_states = dict()
        for var in self.features:
            self.past_states[var + '_h'] = tf.zeros([self.units, 1])
            self.past_states[var + '_c'] = tf.zeros([self.units, 1])
            self.new_states[var + '_h'] = tf.zeros([self.units, 1])
            self.new_states[var + '_c'] = tf.zeros([self.units, 1])

        create_folder(self.folder)

        input_layer = Input(shape=(self.n_past, self.n_features))

        head_list = []
        for var in self.features:           
            # Input attention
            if use_attention_layer:
                ca = CAttention(self.units,
                                self.batch_size,
                                np.array(causal_matrix[features.index(var), :]), 
                                name = var + '_CA')(input_layer, 
                                                    self.past_states, 
                                                    self.past_states)
                # ca = CAttention(self.units[0],
                #                 self.batch_size,
                #                 np.array(causal_matrix[features.index(var), :]), 
                #                 name = var + '_CA')(input_layer, 
                #                                     concatenate([self.past_states[var + '_h'] for var in self.features], axis = 1), 
                #                                     concatenate([self.past_states[var + '_c'] for var in self.features], axis = 1))
            
            # Encoder
            lstm_enc, hidden, cell = LSTM(self.units, 
                                          name = str(var) + '_ENC',
                                          return_sequences = False,
                                          return_state = True,
                                          input_shape = (self.n_past, self.n_features))(ca if use_attention_layer else input_layer)
            self.past_states[var + '_h'] = tf.transpose(hidden)
            self.past_states[var + '_c'] = tf.transpose(cell)
            
            repeat = RepeatVector(self.n_future, name = str(var) + '_REPEAT')(lstm_enc)

            # Decoder
            lstm_dec = LSTM(self.units, 
                            name = str(var) + '_DEC',
                            return_sequences = True)(repeat)
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
        history = self.model.fit(x = trainX, y = trainY, validation_data = (valX, valY), batch_size = self.batch_size,
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