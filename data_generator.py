import numpy as np
import random
import math        


class Data_Generator():

    def __init__(self, n_size, categories, number_of_categories, pictures_per_category, train_val_test_percent, centered_percent):
        self.n_size = n_size
        #numpy array of shape = (number_of_categories, number_of_categories, 1)  example [[1 0 0 0]
        #                                                                                 [0 1 0 0]
        #                                                                                 [0 0 1 0]
        #                                                                                 [0 0 0 1]]
        # can use identity matrix of size number_of_categories 
        self.categories = categories
        self.number_of_categories = number_of_categories
        self.pictures_per_category = pictures_per_category
        self.centered_percent = centered_percent
        self.total_images = self.pictures_per_category * self.number_of_categories
        self.train_size = math.ceil( train_val_test_percent[0] * self.total_images)
        self.valid_size = math.ceil(train_val_test_percent[1] * self.total_images)
        self.test_size = math.ceil(train_val_test_percent[2] * self.total_images)


    def generate_dataset(self):
        images = np.zeros(shape=(self.total_images,self.n_size* self.n_size))
        labels = np.zeros(shape=(self.total_images, self.number_of_categories))
        counter = 0
        for category in self.categories:
            for picture in range(self.pictures_per_category):
                if category.all() == self.categories[0].all():
                    images[counter] += self.generate_random_horizontal_bar_image(True)
                    labels[counter] += category
                    counter += 1
                elif category == self.categories[1]:
                    images[counter] += self.generate_random_vertical_bar_image(True)
                    labels[counter] += category
                    counter += 1
                elif category == self.categories[2]:
                    images[counter] += self.generate_random_circle_image(True)
                    labels[counter] += category
                    counter += 1
                elif category == self.categories[3]:
                    images[counter] += self.generate_random_rectangle_image(True)
                    labels[counter] += category
                    counter += 1
        #Generate a numpy array of evenly spaced indexes, then shuffle
        indexes = np.arange(self.total_images)
        np.random.shuffle(indexes)
        x_train, y_train = images[indexes[:self.train_size]] , labels[indexes[:self.train_size]]
        x_validate , y_validate = images[indexes[self.train_size:self.valid_size]] , labels[indexes[self.train_size:self.valid_size]]
        x_test , y_test = images[indexes[self.valid_size:]] , labels[indexes[self.valid_size:]]
        
        return x_train, y_train, x_validate, y_validate, x_test, y_test

    def generate_random_horizontal_bar_image(self, flat):
        return np.random.rand(self.n_size * self.n_size)

    def generate_random_vertical_bar_image(self, flat):
        return np.random.rand(self.n_size * self.n_size)

    def generate_random_circle_image(self, flat):
        return np.random.rand(self.n_size * self.n_size)

    def generate_random_rectangle_image(self, flat):
        return np.random.rand(self.n_size * self.n_size)