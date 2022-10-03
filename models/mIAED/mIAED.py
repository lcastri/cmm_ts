from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model
from models.MyModel import MyModel
from .IAED import IAED
from models.words import *


class mIAED(MyModel):
    def __init__(self, config, loss, optimizer, metrics):
        super().__init__(config)
        self.channels = dict()
        for var in self.config[W_SETTINGS][W_FEATURES]: self.channels[var] = IAED(self.config, var, name = var + "_IAED")
        self.model = self.create_model(loss, optimizer, metrics)
        plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)
        


    def create_model(self, loss, optimizer, metrics) -> Model:
        inp = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
        
        # Multihead
        multichannels = list()
        for var in self.config[W_SETTINGS][W_FEATURES]:
            multichannels.append(self.channels[var](inp))

        # Concatenation
        y = concatenate(multichannels, axis = 2)
    
        m = Model(inp, y)
        m.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        m.summary()
        return m