import numpy as np

def linear(self, x):
    return x

def relu(self, x):
    return max(0, x)

def tanh(self,x):
    return np.tanh(x)

def sigmoid(self, x):
    return 1/(1+ np.exp(-x))

def linear_derivative(self, x):
    return 1

def relu_derivative(self, x):
    if x < 0:
        return 0
    return 1

def tanh_derivative(self, x):
    return 1 - np.tanh(x)**2

def sigmoid_derivative(self, x):
    return sigmoid(self, x)*(1-sigmoid(self, x))

def get_activation_function(name):
    if name == "linear":
        return linear
    elif name == "relu":
        return relu
    elif name == "sigmoid":
        return sigmoid
    elif name == "tanh":
        return tanh
    elif name == "softmax":
        return softmax

def get_activation_derivative(name):
    if name == "linear":
        return linear_derivative
    elif name == "relu":
        return relu_derivative
    elif name == "sigmoid":
        return sigmoid_derivative
    elif name == "tanh":
        return tanh_derivative
    elif name == "softmax":
        return compute_j_soft

def softmax(self, N):
    exp_fun = lambda x: np.exp(x)
    vectorized_exp_fun = np.vectorize(exp_fun)
    #shift every element x of N to avoid NaN values when x >> 0
    N = N - np.max(N)
    S = vectorized_exp_fun(N)
    S = N / sum(N)
    return S

def compute_j_soft(self, S):
    S = np.squeeze(S)
    # Create a mask for Jsoft when i != j and i == j
    j_soft = np.ones(S.shape)
    id_matrix = np.identity(len(S))
    j_soft = j_soft - id_matrix

    #  when i == j, elements x == Si - Si**2
    diagonal_fun = lambda x : x - x**2    
    id_matrix = id_matrix * S
    id_matrix = diagonal_fun(id_matrix)

    # when i != j, elements x == -Si*Sj
    for index, x in np.ndenumerate(j_soft):
        j_soft[index] = j_soft[index] * (-1) * S[index[0]] * S[index[1]]

    # add two matricies togheter to get Jsoft
    j_soft = j_soft + id_matrix
    return j_soft
