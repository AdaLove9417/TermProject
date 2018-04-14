import os
import random
import tensorflow as tf
import cv2
import numpy as np
import math
from PIL import Image

class database:
    def __init__(self):
        return

    def read_label(self, file):
        folders = os.path.split(file)
        label = folders[-2]
        return label

    def reshuffle_test(self):
        self.test_perm = np.random.permutation(self.test_size)

    def sample_train(self):
        return self.train_images.take(np.random.choice(self.train_size, self.batch_size), 0),\
               self.train_labels.take(np.random.choice(self.train_size, self.batch_size), 0)

    def sample_test(self):
        return self.test_images.take(np.random.choice(self.test_size, self.batch_size), 0),\
               self.test_labels.take(np.random.choice(self.test_size, self.batch_size), 0)

    def sample_full_test(self):
        if (self.test_sample_counter == self.test_sample_end):
            self.reshuffle_test()
            self.test_sample_counter = 0
        sample_vals = range(self.test_sample_counter * self.batch_size,
                            (self.test_sample_counter + 1) * self.batch_size)
        self.test_sample_counter = self.test_sample_counter + 1
        return self.test_images.take(sample_vals, 0), self.test_labels.take(sample_vals, 0)

    def load_set(self, imdb_dir, height, width, batch_size, num_classes):
        self.batch_size = batch_size
        train_dir = os.path.join(imdb_dir, 'train')
        self.train_images = []
        self.train_labels = []
        i = 0
        encoded = np.zeros([1, num_classes])
        for class_label in next(os.walk(train_dir))[1]:
            encoded[0, i] = 1
            for image in os.listdir(os.path.join(train_dir, class_label)):
                self.train_images.append(cv2.imread(os.path.join(train_dir, class_label, image)))
                self.train_labels.append(np.copy(encoded))
            encoded[0, i] = 0
            i =  i + 1
        test_dir = os.path.join(imdb_dir, 'test')
        self.test_images = []
        self.test_labels = []
        i = 0
        encoded = np.zeros([1, num_classes])
        for class_label in next(os.walk(test_dir))[1]:
            encoded[0, i] = 1
            for image in os.listdir(os.path.join(test_dir, class_label)):
                self.test_images.append(cv2.imread(os.path.join(test_dir, class_label, image)))
                self.test_labels.append(np.copy(encoded))
            encoded[0, i] = 0
            i = i + 1
        self.train_images = np.asarray(self.train_images)
        self.train_avg = np.mean(self.train_images, axis=0)
        self.train_labels = np.asarray(self.train_labels)
        self.test_images = np.asarray(self.test_images)
        self.test_labels = np.asarray(self.test_labels)
        self.train_size = len(self.train_labels)
        self.test_size = len(self.test_labels)
        self.reshuffle_test()
        self.test_sample_counter = 0
        self.test_sample_end = math.floor(self.test_size / self.batch_size)