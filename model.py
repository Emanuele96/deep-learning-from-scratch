import numpy as np
import layer

class Model():
    
    def __init__(self, cfg):
        self.layers = list(())

    def add_layer(self,  input_size, output_size, weight_init_range):
        self.layers.append(layer.FC_layer(input_size, output_size, weight_init_range))

    def add_activation(self, size, activation_function):
        self.layers.append(layer.activation_layer(size, activation_function))

    def train(self):
        return -1

    def predict(self):
        return -2