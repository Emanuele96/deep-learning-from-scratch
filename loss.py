import numpy as np

def mse(self, label, output):
    return np.mean(np.power(label - output , 2))

def cross_entropy(self, label, output):
    return -1

def mse_derivative(self, label, output):
    n = len(label)
    return (label - output)*(2/n)

def cross_entropy_derivative(self, label, output):
    return -1
    
def get_loss_function(name):
    if name == "mse":
        return mse
    elif name == "cross_entropy":
        return cross_entropy

def get_loss_derivative(name):
    if name == "mse":
        return mse_derivative
    elif name == "cross_entropy":
        return cross_entropy_derivative