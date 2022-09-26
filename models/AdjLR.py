import keras
import tensorflow as tf

class AdjLR(keras.callbacks.Callback):
    def __init__ (self, model, freq, factor, verbose):
        self.model=model
        self.freq=freq
        self.factor =factor
        self.verbose=verbose
        self.adj_epoch=freq
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.adj_epoch: # adjust the learning rate
            lr=float(tf.keras.backend.get_value(self.model.optimizer.lr)) # get the current learning rate
            new_lr=lr * self.factor
            self.adj_epoch +=self.freq
            if self.verbose == 1:
                print('\non epoch ',epoch + 1, ' lr was adjusted from ', lr, ' to ', new_lr)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr) # set the learning rate in the optimizer