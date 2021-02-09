import numpy as np
import activations

class FC_layer():
    def __init__(self, input_size, output_size, weight_init_range):
  
        self.output = np.zeros(output_size)
        self.weights = np.random.uniform(low=weight_init_range[0], high= weight_init_range[1], size=(output_size, input_size))
        self.bias = np.random.rand(output_size)

    def forward(self):
        return -1
    
    def backward(self):
        return -1

class activation_layer():
    def __init__(self, size, activation_fun):
        self.output = np.zeros(size)
        self.activation_function = activations.get_activation_function(activation_fun)
        self.activation_derivative = activations.get_activation_derivative(activation_fun)
        activations.compute_j_soft(np.random.rand(5))

    def forward(self, input):
        return self.activation_function(input)
    
    def backward(self):
        #takes in the Jacobian matrix Jls (der. loss respect of output) return the Jacobian matrices of the 
        return -1