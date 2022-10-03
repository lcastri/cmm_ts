from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model
from models.MyModel import MyModel
from .T2VRNN import T2VRNN
from models.words import *


class mT2VRNN(MyModel):
    def __init__(self, config : dict = None, folder : str = None):
        super().__init__(config = config, folder = folder)


    def create_model(self, loss, optimizer, metrics) -> Model:
        inp = Input(shape = (self.config[W_SETTINGS][W_NPAST], self.config[W_SETTINGS][W_NFEATURES]))
        
        # Multihead
        channels = list()
        for var in self.config[W_SETTINGS][W_FEATURES]:
            channels.append(T2VRNN(self.config, var, name = var + "_T2VRNN")(inp))

        # Concatenation
        y = concatenate(channels, axis = 2)
    
        m = Model(inp, y)
        m.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        m.summary()

        self.model = m
        plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)
        return m
