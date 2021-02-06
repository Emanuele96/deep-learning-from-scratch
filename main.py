import data_generator
import numpy as np

if __name__ == "__main__":

    categories = np.identity(4, dtype=int)
    data_generator = data_generator.Data_Generator(128, categories, 4, 5, (0.7, 0.2, 0.1), 0.5, 0.05)
    x_train, y_train, x_validate, y_validate, x_test, y_test = data_generator.generate_dataset()
    #print(x_train)
    #print(y_train)
    noise = data_generator.generate_noise(False)
    pic = np.round(data_generator.generate_random_vertical_bar_image(False))
    data_generator.show_picture(pic)
    data_generator.show_picture(data_generator.apply_noise(pic, noise))
