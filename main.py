import model
import data_generator
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    cfg = {
        #Data generator variables
        "n_size" : 16,
        "categories" : np.expand_dims(np.identity(4, dtype=int),1),
        "pic_per_categories" :250,
        "train_val_test_percent" : (0.7, 0.2, 0.1),
        "center_image_prob" : 0.5,
        "noise_percent" : 0.05,
        #Model variables
        "hidden_layers_nodes" : (150,50,4),
        "output_layer_nodes" : 4,
        "use_softmax" : True,
        "lr": 5*10e-4,
        "loss_fun" : "cross_entropy",
        "batch_size" : 16,
        "epochs" : 1000

    }
    data_generator = data_generator.Data_Generator(cfg["n_size"], cfg["categories"], len(cfg["categories"]), cfg["pic_per_categories"], cfg["train_val_test_percent"], cfg["center_image_prob"], cfg["noise_percent"])
    x_train, y_train, x_validate, y_validate, x_test, y_test = data_generator.generate_dataset()

    #noise = data_generator.generate_noise(False)
    #pic = np.round(data_generator.generate_random_rectangle_image(False))
    #data_generator.show_picture(pic)
    #data_generator.show_picture(noise)
    #data_generator.show_picture(data_generator.apply_noise(pic, noise))
    m1 = model.Model(cfg)
    input_size = cfg["hidden_layers_nodes"][0]
    input_size = m1.add_layer(cfg["n_size"]**2,input_size, (-0.1, 0.1), "sigmoid")
    input_size = m1.add_layer(input_size, cfg["hidden_layers_nodes"][1], (-0.1, 0.1), "sigmoid")
    input_size = m1.add_layer(input_size, cfg["hidden_layers_nodes"][2], (-0.1, 0.1), "sigmoid")
    if cfg["use_softmax"]:
        m1.add_softmax()  
    print(m1)
    losses = m1.train(x_train, y_train, cfg["epochs"], cfg["batch_size"])
    mini_batches = np.linspace(0, len(losses), num=len(losses))
    plt.plot(mini_batches, losses)
    plt.show()