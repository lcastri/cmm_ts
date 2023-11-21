from keras.layers import *
from keras.models import *
# from keras.utils.vis_utils import plot_model
from models.utils import Models
from models.MyModel import MyModel
from .IAED2 import IAED
import models.Words as W


class sIAED(MyModel):
    def __init__(self, df, config : dict = None, folder : str = None):
        super().__init__(name = Models.sIAED, df = df, config = config, folder = folder)
               

    def create_model(self, target_var, loss, optimizer, metrics, searchBest = False) -> Model:
        self.target_var = target_var

        inp = Input(shape = (self.config[W.NPAST], self.config[W.NFEATURES]))
        x = IAED(self.config, target_var, name = target_var + "_IAED", searchBest = searchBest)(inp)
    
        self.model = Model(inp, x)
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        self.model.summary()
        # plot_model(self.model, to_file = self.model_dir + '/model_plot.png', show_shapes = True, show_layer_names = True, expand_nested = True)
        return self.model