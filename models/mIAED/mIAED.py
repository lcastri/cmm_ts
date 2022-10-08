from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model
from models.MyModel import MyModel
from .IAED import IAED
from models.utils import Words as W


class mIAED(MyModel):
    def __init__(self, config : dict = None, folder : str = None):
        super().__init__(config = config, folder = folder)        


    def create_model(self, loss, optimizer, metrics) -> Model:
        inp = Input(shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
        
        # Multihead
        channels = list()
        for var in self.config[W.FEATURES]:
            channels.append(IAED(self.config, var, name = var + "_IAED")(inp))

        # Concatenation
        y = concatenate(channels, axis = 2)
    
        m = Model(inp, y)
        m.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        m.summary()

        self.model = m
        # plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)
        return m