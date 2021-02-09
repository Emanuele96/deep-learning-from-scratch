import numpy as np
import activations

class FC_layer():
    def __init__(self, input_size, output_size, weight_init_range):
        self.input = None
        self.output = None
        self.weights = np.random.uniform(low=weight_init_range[0], high= weight_init_range[1], size=(output_size, input_size))
        self.bias = np.random.rand(output_size)

    def forward(self, input_activations):
        # Dot product of input with W plus bias. Cache and return
        output = np.dot(input_activations, self.weights) + self.bias
        self.input_activations = input_activations
        self.output = output
        return self.output
    
    def backward(self):
        return -1

class activation_layer():
    def __init__(self, size, activation_fun):
        self.type = activation_fun
        self.output = np.zeros(size)
        self.activation_function = activations.get_activation_function(activation_fun)
        self.activation_derivative = activations.get_activation_derivative(activation_fun)

    def forward(self, input_data):
        return self.activation_function(self, input_data)
    
    def backward(self, input_data):
        #takes in the Jacobian matrix Jls (der. loss respect of output) return the Jacobian matrix Jsoft.
        # Do the dot product of the Jsoft and Jls to get Jlz, to be fed to the rest of the network
        if self.type == "softmax":
            return np.dot(input_data, self.activation_derivative(self, input_data))
        # If regular activation, we do the element wise multiplication between two row vectors:
        #   the input data and the [F'(inputdata)] to get the layer input
        return np.multiply(self.activation_derivative(self, input_data), input_data)