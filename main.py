import model
import data_generator
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    cfg = {
        #Data generator variables
        "n_size" : 6,
        "categories" : np.expand_dims(np.identity(4, dtype=int),1),
        "pic_per_categories" :10,
        "train_val_test_percent" : (0.7, 0.2, 0.1),
        "center_image_prob" : 0.5,
        "noise_percent" : 0.05,
        #Model variables
        "hidden_layers_nodes" : (20,15,10,5),
        "output_layer_nodes" : 4,
        "use_softmax" : True,
        "lr": 0.5

    }
    data_generator = data_generator.Data_Generator(cfg["n_size"], cfg["categories"], len(cfg["categories"]), cfg["pic_per_categories"], cfg["train_val_test_percent"], cfg["center_image_prob"], cfg["noise_percent"])
    x_train, y_train, x_validate, y_validate, x_test, y_test = data_generator.generate_dataset()

    #noise = data_generator.generate_noise(False)
    #pic = np.round(data_generator.generate_random_circle_image(False))
    #data_generator.show_picture(pic)
    #data_generator.show_picture(data_generator.apply_noise(pic, noise))
    m1 = model.Model(cfg)
    m1.add_layer(36,100, (-0.1, 0.1), "relu")
    m1.add_layer(100,4, (-0.1, 0.1), "relu")
    m1.add_activation("softmax")  
    for layer in m1.layers:
        print(layer)

    for i in range(10):
        in_data =x_train[i]
        for layer in m1.layers:
            in_data = layer.forward(in_data)
        print(in_data)
    losses = m1.train(x_train, y_train, 10, 5)
    mini_batches = np.linspace(0, len(losses), num=len(losses))
    print(losses)
    #plt.plot(mini_batches, losses)