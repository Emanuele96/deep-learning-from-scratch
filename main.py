import data_generator
import numpy as np

if __name__ == "__main__":

    categories = np.identity(4, dtype=int)
    data_generator = data_generator.Data_Generator(3, categories, 4, 5, (0.7, 0.2, 0.1), 0.8)
    x_train, y_train, x_validate, y_validate, x_test, y_test = data_generator.generate_dataset()
    print(x_train)
    print(y_train)