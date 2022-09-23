import pickle
from keras.layers import *
from keras.models import *
from old.IAED import IAED
from matplotlib import pyplot as plt

class mIAED(Model):
    def __init__(self, folder, enc_dec_units, out_units, features, n_past, n_feature, n_future, 
                 use_attention, use_causality, causal_matrix):
        super(mIAED, self).__init__()
        self.folder = folder
        self.enc_dec_units = enc_dec_units
        self.out_units = out_units
        self.features = features
        self.n_past = n_past
        self.n_feature = n_feature
        self.n_future = n_future
        self.use_attention = use_attention
        self.use_causality = use_causality
        self.causal_matrix = causal_matrix
        self.channels = dict()

        # Multihead
        for var in self.features:
            self.channels[var] = IAED(self.enc_dec_units, self.out_units, n_past, n_feature, n_future,
                                      var, use_attention, use_causality,
                                      causal_matrix[self.features.index(var), :],
                                      name = var + "_IAED")

        # Concatenation
        self.concat = Concatenate()
    

    def call(self, input_sequence):
        resMultiHead = list()
        x = input_sequence

        for var in self.features:
            resMultiHead.append(self.channels[var](x))
        
        concat = self.concat(resMultiHead)

        return concat

    
    def model(self):
        x = Input(shape = (self.n_past, self.n_feature))
        return Model(inputs = [x], outputs = self.call(x))


    def get_config(self):
        data = dict()
        if self.use_attention and self.use_causality:
            for var in self.features:
                data[var] = self.channels[var].ca.causal.numpy()
        return {"causal_weights": data}


    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        history = super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
        
        plt.figure()
        plt.plot(history.history["loss"], label = "Training loss")
        plt.plot(history.history["val_loss"], label = "Validation loss")
        plt.legend()
        plt.savefig(self.folder + "/plots/loss.png", dpi = 300)
        plt.savefig(self.folder + "/plots/loss.eps", dpi = 300)

        plt.figure()
        plt.plot(history.history["mae"], label = "Training mae")
        plt.plot(history.history["val_mae"], label = "Validation mae")
        plt.legend()
        plt.savefig(self.folder + "/plots/mae.png", dpi = 300)
        plt.savefig(self.folder + "/plots/mae.eps", dpi = 300)

        plt.figure()
        plt.plot(history.history["mape"], label = "Training mape")
        plt.plot(history.history["val_mape"], label = "Validation mape")
        plt.legend()
        plt.savefig(self.folder + "/plots/mape.png", dpi = 300)
        plt.savefig(self.folder + "/plots/mape.eps", dpi = 300)

        plt.figure()
        plt.plot(history.history["accuracy"], label = "Training accuracy")
        plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
        plt.legend()
        plt.savefig(self.folder + "/plots/accuracy.png", dpi = 300)
        plt.savefig(self.folder + "/plots/accuracy.eps", dpi = 300)

        with open(self.folder + '/history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
