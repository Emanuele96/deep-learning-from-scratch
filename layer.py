import numpy as np
import activations

class FC_layer():
    def __init__(self, input_size, output_size, weight_init_range):
        self.type = "FC"
        self.shape = (input_size, output_size)
        self.input = None
        self.output = None
        self.weights = np.random.uniform(low=weight_init_range[0], high= weight_init_range[1], size=(input_size, output_size))
        self.bias = np.random.rand(output_size)
        self.weights_grads = np.zeros(self.weights.shape)
        self.bias_grads = np.zeros(self.bias.shape)
        

    def forward(self, input_activations):
        # Dot product of input with W plus bias. Cache and return

        output = np.dot(input_activations, self.weights) + self.bias
        self.input_activations = input_activations
        self.input = input_activations
        self.output = output
        return self.output
    
    def backward(self, jacobian_L_Z):
        # Get the output_loss of this layer from the activation layer.
        # Calculate the weights gradients, the bias gradient and the input_loss
        #  that will be passed to the previous activation layer and so on, up to layer 1
        Y = self.input
        jacobian_Z_sum = self.create_jacobian_Z_sum()
        # Weights loss derivative --> Layer activation input transposed (to get column vector) dot product with output loss
        simp_jacobian_Z_W = np.outer(Y, jacobian_Z_sum.diagonal())

        jacobian_L_W = jacobian_L_Z * simp_jacobian_Z_W
        # Calculate the input layer loss
        # by doing dot product of output layer loss and the weigths matrix transposed (so to invert M N to N M, where M < N, we go the other way around)
        jacobian_Z_Y = np.dot(jacobian_Z_sum ,self.weights.T)
        jacobian_L_Y = np.dot( jacobian_L_Z, jacobian_Z_Y)
        

        # Bias loss is the as the output loss --> the bias influence on the loss == layer activation output influence on the loss
        jacobian_L_B = jacobian_L_Z

      

        # Now save the bias loss and weight loss (representing the calculated gradiants).
        # This will be updated at the end of the batch, or SGD
        self.weights_grads = self.weights_grads + jacobian_L_W
        self.bias_grads = self.bias_grads + jacobian_L_B

        #Finally return the calculated input loss --> this will be the output loss of the next layer
        return jacobian_L_Y

    def create_jacobian_Z_sum(self):
        Z = np.squeeze(self.output)
        n = len(Z)
        jacobian_Z_sum = np.zeros((n, n))
        for index, x in np.ndenumerate(jacobian_Z_sum):
            if index[0] == index[1]:
                z =  Z[index[0]]
                jacobian_Z_sum[index] = z * (1 - z)
        return jacobian_Z_sum

    def update_gradients(self, learning_rate, gradient_avg_factor = 1):
        #Update gradients, usefull when doing batch learning
        # Get the avg of the gradients (for SGD divide by 1, else divide by batchsize)
        self.weights_grads = self.weights_grads / gradient_avg_factor
        self.bias_grads = self.bias_grads / gradient_avg_factor

        # Update weights and biases
        self.weights = self.weights - learning_rate * self.weights_grads
        self.bias = self.bias - learning_rate * self.bias_grads
        self.weights_grads = np.zeros(self.weights.shape)
        self.bias_grads = np.zeros(self.bias.shape)


    def __str__(self):
        return "FC Layer type size = " + str(self.weights.shape)

class activation_layer():
    def __init__(self, size, activation_fun):
        self.size = size
        self.type = activation_fun
        self.output = np.zeros(size)
        self.activation_function = activations.get_activation_function(activation_fun)
        self.activation_derivative = activations.get_activation_derivative(activation_fun)

    def forward(self, input_data):
        return self.activation_function(self, input_data)
    
    def backward(self, jacobian_L_Y):
        '''  #takes in the Jacobian matrix Jls (der. loss respect of output) return the Jacobian matrix Jsoft.
        # Do the dot product of the Jsoft and Jls to get Jlz, to be fed to the rest of the network
        if self.type == "softmax":
            return np.dot(output_loss, activations.compute_j_soft(self, network_output))
        ''' 
        # If regular activation, we do the element wise multiplication between two row vectors:
        #   the input data and the [F'(inputdata)] to get the layer input
        return self.activation_derivative(self, jacobian_L_Y)

    def __str__(self):
        return "Activation Layer type " + self.type.upper() + " size = " + str(self.output.shape[0])
