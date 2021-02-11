import numpy as np
import layer
import loss
import activations
import math
from progress.bar import IncrementalBar


class Model():
    
    def __init__(self, cfg):
        self.layers = list(())
        self.learning_rate = cfg["lr"]
        self.loss_fun = loss.get_loss_function(cfg["loss_fun"])
        self.loss_derivative = loss.get_loss_derivative(cfg["loss_fun"])

    def add_layer(self,  input_size, output_size, weight_init_range, activation_function):
        self.layers.append(layer.FC_layer(input_size, output_size, weight_init_range))
        self.layers.append(layer.activation_layer(output_size, activation_function))
        return output_size

    def add_activation(self, activation_function):
        self.layers.append(layer.activation_layer(self.layers[-1].size, activation_function))

    def train(self, x_train, y_train, epochs, batch_size):
        bar = IncrementalBar('Training epoch', max=epochs)
        samples = len(x_train)
        batches = math.ceil(samples/batch_size)
        losses = list(())
        for i in range(epochs):
            # For each epoch --> go through the whole train set, one batch on the time
            for j in range(batches):
                # For each batch, go through "batch_size" samples 
                batch_loss = 0
                for k in range(batch_size):
                    # For each sample, propagate to the network.
                    # Then backpropagate network output and calculate gradients.
                    sample_nr = k + j * batch_size
                    if sample_nr == samples:
                        break
                    

                    # FORWARD PASS : Fetch the input data and propagate through the network
                    # This will be represent the input layer
                    network_output = x_train[sample_nr]
                    for layer in self.layers:
                        network_output = layer.forward(network_output)
                    
                    # Calculate the loss
                    loss = self.loss_fun(self,y_train[sample_nr], network_output)
                    batch_loss = batch_loss + loss
                    #print(loss)
                    #print("network output", network_output)
                    #print("label", y_train[sample_nr])
                    #print("loss", loss)
                    
                    # BACKWARD PASS : Get the loss and backpropagate throught the network
                    # If the last layer is Softmax:
                    #   Calculate JLS and JSZ (Jsoft).
                    #   Then calculate the jacobian of input prior the Softmax layer, JLZ
                    #   Backpropagate JLZ to all the other layers.
                    if self.layers[-1].type == "softmax":
                        jacobian_L_S = self.loss_derivative(self,y_train[sample_nr],  network_output)
                        Jacobian_soft = activations.compute_j_soft(self, network_output)
                        jacobian_L_Z = np.dot(jacobian_L_S, Jacobian_soft)
                        layers = self.layers[:-1]
                    else:
                        jacobian_L_Z = self.loss_derivative(self, y_train[sample_nr], network_output)
                        layers = self.layers

                    for layer in reversed(layers):
                        jacobian_L_Z = layer.backward(jacobian_L_Z)

                # At the end of each batch, update gradients
                for layer in self.layers:
                    if layer.type == "FC":
                        layer.update_gradients(self.learning_rate)
                losses.append(batch_loss/batch_size)
            # end of an epoch
            bar.next()
        bar.finish()
        return losses

    def predict(self):
        return -2

    def __str__(self):
        s = "***  Model Architecture *** \n Input Layer of size = " + str(self.layers[0].shape[0])
        for layer in self.layers:
            s = s + "\n" + str(layer)
        s = s + "\n" + "**************************"
        return s
