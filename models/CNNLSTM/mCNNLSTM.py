from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model
from constants import LIST_FEATURES
from models.MyModel import MyModel
from .CNNLSTM import CNNLSTM
from models.utils import Models 
import models.Words as W


class mCNNLSTM(MyModel):
    def __init__(self, df, config : dict = None, folder : str = None):
        super().__init__(name = Models.mCNN, df = df, config = config, folder = folder)        


    def create_model(self, loss, optimizer, metrics, searchBest = False) -> Model:
        inp = Input(shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
        
        # Multihead
        channels = list()
        list_f = LIST_FEATURES if searchBest else self.config[W.FEATURES]
        for var in list_f:
            channels.append(CNNLSTM(self.config, var, name = var + "_CNNLSTM", searchBest = searchBest)(inp))
        
        # Concatenation
        y = concatenate(channels, axis = 2)
    
        self.model = Model(inp, y)
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        self.model.summary()

        # plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)
        return self.model