import numpy as np
import activations

class FC_layer():
    def __init__(self, input_size, output_size, weight_init_range, activation):
        self.type = "FC"
        self.shape = (input_size, output_size)
        self.activation = activations.get_activation_function(activation)
        self.d_activation = activations.get_activation_derivative(activation)
        self.input = None
        self.output = None
        self.weights = np.random.uniform(low=weight_init_range[0], high= weight_init_range[1], size=(input_size, output_size))
        self.bias = np.zeros((1,output_size))
        self.weights_grads = np.zeros(self.weights.shape)
        self.bias_grads = np.zeros(self.bias.shape)
        

    def forward(self, input_activations):
        # Dot product of input with W plus bias. Cache, activate and return
        output = np.dot(input_activations, self.weights) + self.bias
        output = self.activation(self, output)
        self.output = output
        self.input = input_activations
        return output
    
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
        self.weights_grads =self.weights_grads + jacobian_L_W
        self.bias_grads = self.bias_grads + jacobian_L_B
        #Finally return the calculated input loss --> this will be the output loss of the next layer
        return jacobian_L_Y

    def create_jacobian_Z_sum(self):
        return np.identity(self.output[0].size) * self.d_activation(self, self.output)

    def update_gradients(self, learning_rate, gradient_avg_factor = 1):
        #Update gradients, usefull when doing batch learning
        # Get the avg of the gradients (for SGD divide by 1, else divide by batchsize)
        #self.weights_grads = self.weights_grads / gradient_avg_factor
        #self.bias_grads = self.bias_grads / gradient_avg_factor

        # Update weights and biases
        self.weights -= learning_rate * self.weights_grads
        self.bias -= learning_rate * self.bias_grads
        self.weights_grads = np.zeros(self.weights.shape)
        self.bias_grads = np.zeros(self.bias.shape)


    def __str__(self):
        return "FC Layer type size = " + str(self.weights.shape)

class softmax():
    def __init__(self, size):
        self.size = size
        self.type = "softmax"
        self.activation_function = activations.softmax

    def forward(self, input_data):
        return  self.activation_function(self, input_data)

    def backward(self, jacobian_L_S, softmaxed_network_output):
        jacobian_soft = self.compute_j_soft(softmaxed_network_output)    
        jacobian_L_Z = np.dot(jacobian_L_S, jacobian_soft)
        return jacobian_L_Z

    def compute_j_soft(self, S):
        S = np.squeeze(S)
        n = len(S)
        j_soft = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    j_soft[i][j] = S[i] - S[i]**2
                else:
                    j_soft[i][j] = -S[i]*S[j]
        return j_soft

    def __str__(self):
        return "Softmax Layer of size = " + str(self.size)
