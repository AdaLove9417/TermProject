import os
import random
import tensorflow as tf
import cv2
from PIL import Image

class database:
    def __init__(self):
        return

    def read_label(self, file):
        folders = os.path.split(file)
        label = folders[-2]
        return label

    def load_set(self, imdb_dir, height, width, batch_size):
        train_dir = os.path.join(imdb_dir, 'train')
        train_images = []
        train_labels = []
        for class_label in next(os.walk(train_dir))[1]:

            for image in os.listdir(os.path.join(train_dir, class_label)):
                train_images.append(os.path.join(train_dir, class_label, image))
                train_labels.append(class_label)
        test_dir = os.path.join(imdb_dir, 'test')
        test_images = []
        test_labels = []
        for class_label in next(os.walk(test_dir))[1]:
            for image in os.listdir(os.path.join(test_dir, class_label)):
                test_images.append(os.path.join(test_dir, class_label, image))
                test_labels.append(class_label)
        train_input_queue = tf.train.slice_input_producer(
            [train_images, train_labels])
        test_input_queue = tf.train.slice_input_producer(
            [test_images, test_labels])

        # process path and string tensor into an image and a label
        file_content = tf.read_file(train_input_queue[0])
        train_image = tf.image.decode_jpeg(file_content, channels=3)
        train_label = train_input_queue[1]

        file_content = tf.read_file(test_input_queue[0])
        test_image = tf.image.decode_jpeg(file_content, channels=3)
        test_label = test_input_queue[1]

        train_image.set_shape([height, width, 3])
        test_image.set_shape([height, width, 3])

        self.train_image_batch, self.train_label_batch = tf.train.batch(
            [train_image, train_label],
            batch_size=batch_size)
        self.test_image_batch, self.test_label_batch = tf.train.batch(
            [test_image, test_label],
            batch_size=batch_size)