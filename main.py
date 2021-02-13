import model
import data_generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from progress.bar import IncrementalBar
import json
import argparse


'''cfg = {
    #Data generator variables
    "n_size" : 16,
    "categories" : 4,
    "pic_per_categories" :150,
    "train_val_test_percent" : (0.7, 0.2, 0.1),
    "center_image_prob" : 0.5,
    "noise_percent" : 0.05,
    "animation_speed" : 50,
    "show_pictures_on_start": False,
    #Model variables
    "train_on_start": True,
    "hidden_layers_nodes" : (20,10),
    "hidden_layers_activations":("tanh", "tanh"),
    "hidden_layers_weight_start":((-0.1, 0.1),(-0.1, 0.1)),
    "output_layer_nodes" : 4,
    "output_layer_activation": "tanh",
    "output_layer_weight_start": (-0.1, 0.1),
    "use_softmax" : True,
    "lr": 5e-03,#9e-04,
    "loss_fun" : "cross_entropy",
    "batch_size" : 32,
    "epochs" : 300
}

cfg33 = {
    #Data generator variables
    "n_size" : 16,
    "categories" : 4,
    "pic_per_categories" :150,
    "train_val_test_percent" : (0.7, 0.2, 0.1),
    "center_image_prob" : 0.5,
    "noise_percent" : 0.05,
    "animation_speed" : 50,
    "show_pictures_on_start": False,
    #Model variables
    "train_on_start": True,
    "hidden_layers_nodes" : (100,100,10),
    "hidden_layers_activations":("relu", "relu", "relu"),
    "hidden_layers_weight_start":((-0.1, 0.1),(-0.1, 0.1),(-0.1, 0.1)),
    "output_layer_nodes" : 4,
    "output_layer_activation": "relu",
    "output_layer_weight_start": (-0.1, 0.1),
    "use_softmax" : True,
    "lr": 5e-03,#9e-04,
    "loss_fun" : "cross_entropy",
    "batch_size" : 32,
    "epochs" : 300
}

'''
def read_config_from_json(filename):
    with open(filename, 'r') as fp:
        cfg = json.load(fp)
    return cfg

def dump_config_to_json(filename):
    with open(filename, 'w') as fp:
        json.dump(cfg, fp)

def animate(i):
    plt.clf()
    if i == len(x_train):
        print( '.', end ='' ) 
        bar.finish()
    im = plt.imshow(np.reshape(x_train[i], (cfg["n_size"],cfg["n_size"])),interpolation="nearest")
    bar.next()
    return im

# Parse config file of choice
parser = argparse.ArgumentParser("Deep Learning Project 1")
parser.add_argument('--config', default="cfg.json", type=str, help="Select configuration file to load")
args = parser.parse_args()
cfg = read_config_from_json(args.config)
# Generate dataset
data_generator = data_generator.Data_Generator(cfg["n_size"], cfg["categories"], cfg["categories"], cfg["pic_per_categories"], cfg["train_val_test_percent"], cfg["center_image_prob"], cfg["noise_percent"])
x_train, y_train, x_validate, y_validate, x_test, y_test = data_generator.generate_dataset()

# Run Program
if __name__ == "__main__":

    if cfg["show_pictures_on_start"]:
        bar = IncrementalBar('X_train input', max=len(x_train))
        fig = plt.figure( )
        anim = animation.FuncAnimation(fig, animate, interval  = cfg["animation_speed"])
        plt.show()
    # Construct the model
    m1 = model.Model(cfg)
    input_size = cfg["n_size"]**2
    # Add hidden layers
    for i in range(len(cfg["hidden_layers_nodes"])):
        input_size = m1.add_layer(input_size, cfg["hidden_layers_nodes"][i], cfg["hidden_layers_weight_start"][i], cfg["hidden_layers_activations"][i])
    # Add output layer
    m1.add_layer(input_size, cfg["output_layer_nodes"], cfg["output_layer_weight_start"], cfg["output_layer_activation"])
    # Add Softmax if required
    if cfg["use_softmax"]:
        m1.add_softmax()  
    print(m1)
    #Train the model
    if cfg["train_on_start"]:
        losses, validation_errors = m1.train(x_train, y_train, x_validate, y_validate, cfg["epochs"], cfg["batch_size"])
        mini_batches = np.linspace(0, len(losses), num=len(losses))
        plt.plot(mini_batches, losses)
        plt.plot(mini_batches, validation_errors)
        plt.show()
    


