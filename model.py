import numpy as np
import layer

class Model():
    
    def __init__(self, cfg):
        self.layers = list(())

    def add_layer(self, input_size, output_size, activation_fun, weight_init_range):
        self.layers.append(layer.Layer(input_size, output_size, activation_fun, weight_init_range))

    def train(self):
        return -1

    def predict(self):
        return -2