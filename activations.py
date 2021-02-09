import numpy as np

def linear(x):
    return x

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def linear_derivative():
    return 1

def relu_derivative(x):
    if x < 0:
        return 0
    return 1

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def get_activation_function(name):
    if name == "linear":
        return linear
    elif name == "relu":
        return relu
    elif name == "sigmoid":
        return sigmoid
    elif name == "tanh":
        return tanh

def get_activation_derivative(name):
    if name == "linear":
        return linear_derivative
    elif name == "relu":
        return relu_derivative
    elif name == "sigmoid":
        return sigmoid_derivative
    elif name == "tanh":
        return tanh_derivative
