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

def softmax(N):
    exp_fun = lambda x: np.exp(x)
    vectorized_exp_fun = np.vectorize(exp_fun)
    #shift every element x of N to avoid NaN values when x >> 0
    N = N - np.max(N)
    S = vectorized_exp_fun(N)
    S = N / sum(N)
    return S

def compute_j_soft(S):
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
