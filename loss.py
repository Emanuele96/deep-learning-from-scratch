import numpy as np

def mse(self, label, output):
    return np.mean(np.power(label - output , 2))

def cross_entropy(self, label, output):
    loss = np.zeros(output.shape)
    for y in range (len(label[0])):
        if label[0][y] == 0.9:
            loss[0][y] = - np.log(output[0][y])
        else:
             loss[0][y] = - np.log(1-  output[0][y])
    return np.sum(loss) 

def mse_derivative(self, label, output):
    n = len(label)
    return (label - output)*(2/n)

def cross_entropy_derivative(self, label, output):
    loss = np.zeros(output.shape)
    for y in range (len(label[0])):
        if label[0][y] == 0.9:
            loss[0][y] = -1/output[0][y]
        else:
             loss[0][y] = -1/(1-  output[0][y])
    return loss 
    
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