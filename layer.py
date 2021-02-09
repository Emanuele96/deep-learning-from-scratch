import numpy as np
import activations

class Layer():
    def __init__(self, input_size, output_size, activation_fun, weight_init_range):
  
        self.output = np.zeros(output_size)
        self.weights = np.random.uniform(low=weight_init_range[0], high= weight_init_range[1], size=(output_size, input_size))
        self.bias = np.random.rand(output_size)
        self.activation_function = activations.get_activation_function(activation_fun)
        self.activation_derivative = activations.get_activation_derivative(activation_fun)
        print(self.weights)

    def forward(self):
        return -1
    
    def backward(self):
        return -1