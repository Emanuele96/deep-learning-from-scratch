import numpy as np
import layer
import math

class Model():
    
    def __init__(self, cfg):
        self.layers = list(())
        self.learning_rate = cfg["lr"]
        self.loss_fun = lambda x, y: x 

    def add_layer(self,  input_size, output_size, weight_init_range, activation_function):
        self.layers.append(layer.FC_layer(input_size, output_size, weight_init_range))
        self.layers.append(layer.activation_layer(output_size, activation_function))

    def add_activation(self, activation_function):
        self.layers.append(layer.activation_layer(self.layers[-1].size, activation_function))

    def train(self, x_train, y_train, epochs, batch_size):
        samples = len(x_train)
        batches = math.ceil(samples/batch_size)
        losses = list(())
        for i in range(epochs):
            # For each epoch --> go through the whole train set, one batch on the time
            for j in range(batches):
                # For each batch, go through "batch_size" samples 
                for k in range(batch_size):
                    # For each sample, propagate to the network.
                    # Then backpropagate network output and calculate gradients.
                    sample_nr = k + j * batch_size
                    if sample_nr == samples:
                        break
                    batch_loss = 0

                    # FORWARD PASS : Fetch the input data and propagate through the network
                    network_output = x_train[sample_nr]
                    for layer in self.layers:
                        network_output = layer.forward(network_output)
                    
                    # Calculate the loss
                    loss = self.loss_fun( y_train[sample_nr], network_output)
                    batch_loss += loss
                    # BACKWARD PASS : Get the loss and backpropagate throught the network
                    for layer in reversed(self.layers):
                        loss = layer.backward(loss)

                # At the end of each batch, update gradients
                for layer in self.layers:
                    if layer.type == "FC":
                        layer.update_gradients(self.learning_rate)

                losses.append(batch_loss/batch_size)
            # end of an epoch
        return losses

    def predict(self):
        return -2