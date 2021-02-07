import data_generator
import numpy as np

if __name__ == "__main__":

    cfg = {
        "n_size" : 16,
        "categories" : np.identity(4, dtype=int),
        "pic_per_categories" :250,
        "train_val_test_percent" : (0.7, 0.2, 0.1),
        "center_image_prob" : 0.5,
        "noise_percent" : 0.05
    }
    data_generator = data_generator.Data_Generator(cfg["n_size"], cfg["categories"], len(cfg["categories"]), cfg["pic_per_categories"], cfg["train_val_test_percent"], cfg["center_image_prob"], cfg["noise_percent"])
    x_train, y_train, x_validate, y_validate, x_test, y_test = data_generator.generate_dataset()

    noise = data_generator.generate_noise(False)
    pic = np.round(data_generator.generate_random_circle_image(False))
    data_generator.show_picture(pic)
    data_generator.show_picture(data_generator.apply_noise(pic, noise))
