import model
import data_generator
import numpy as np

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
        "hidden_layers_nodes" : (20,15,10,5),
        "output_layer_nodes" : 4,
        "use_softmax" : True,

    }
    data_generator = data_generator.Data_Generator(cfg["n_size"], cfg["categories"], len(cfg["categories"]), cfg["pic_per_categories"], cfg["train_val_test_percent"], cfg["center_image_prob"], cfg["noise_percent"])
    x_train, y_train, x_validate, y_validate, x_test, y_test = data_generator.generate_dataset()

    #noise = data_generator.generate_noise(False)
    #pic = np.round(data_generator.generate_random_circle_image(False))
    #data_generator.show_picture(pic)
    #data_generator.show_picture(data_generator.apply_noise(pic, noise))
    m1 = model.Model(cfg)
    m1.add_layer(3,5, (-0.1, 0.1))
    m1.add_activation(5, "tanh") 
    print(m1.layers[1].backward(10))