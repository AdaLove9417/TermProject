import inspect
import os
import numpy as np
import tensorflow as tf
import time
from numpy import matlib
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials


class LeviHassner:
    def __init__(self, scratch=None, trainable=True, dropout=0.5):
        if scratch is None:
            self.data_dict = None
            self.trainable = trainable
            self.dropout = dropout
            self.var_dict = {}
        else:
            path = inspect.getfile(LeviHassner)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "levi_and_hassner.npy")
            self.data_dict = np.load(path, encoding='latin1').item()
            print("npy file loaded")

    def build(self, rgb, lr, numclasses, train_mode=None, pretrained=False):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")

        self.x_ = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.y_ = tf.placeholder(tf.float32, [None, 1, numclasses])

        self.conv1 = self.conv_layer(self.x_, 3, 96, 7, "conv1")
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = self.max_pool(self.relu1, 'pool1')
        self.bn1 = tf.nn.local_response_normalization(self.pool1, 1e-4, beta=.75, bias=2)

        self.conv2 = self.conv_layer(self.bn1, 96, 256, 5, "conv2")
        self.relu2 = tf.nn.relu(self.conv2)
        self.pool2 = self.max_pool(self.relu2, 'pool2')
        self.bn2 = tf.nn.local_response_normalization(self.pool2, 1e-4, beta=.75, bias=2)

        self.conv3 = self.conv_layer(self.bn2, 256, 384, 3, "conv3")
        self.relu3 = tf.nn.relu(self.conv3)
        self.pool3 = self.max_pool(self.relu3, 'pool3')

        self.fc4 = self.fc_layer(self.pool3, 9600, 512, "fc4")
        assert self.fc4.get_shape().as_list()[1:] == [4096]
        self.relu4 = tf.nn.relu(self.fc4)
        if train_mode is not None:
            self.relu4 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu4, self.dropout), lambda: self.relu4)
        elif self.trainable:
            self.relu4 = tf.nn.dropout(self.relu4, self.dropout)

        self.fc5 = self.fc_layer(self.relu4, 512, 512, "fc5")
        self.relu5 = tf.nn.relu(self.fc5)
        if train_mode is not None:
            self.relu5 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu5, self.dropout), lambda: self.relu5)
        elif self.trainable:
            self.relu5 = tf.nn.dropout(self.relu5, self.dropout)

        self.fc7 = self.fc_layer(self.relu5, 512, 8, "fc7")

        self.prob = tf.nn.softmax(self.fc7, name="prob")

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.fc8))
        self.hot = tf.one_hot(tf.argmax(self.prob, 1), 8)
        self.correct_no_cast = tf.not_equal(self.hot, self.y_) #to float true is 0
        self.correct_prediction = tf.cast(self.correct_no_cast, tf.float32)

        self.accuracy = tf.reduce_mean(self.correct_prediction)

        non_freeze = ["fc5", "fc6", "fc7"]

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=lr)
        if pretrained:
            self.train_op = self.optimizer.minimize(self.cross_entropy, var_list=non_freeze)
        else:
            self.train_op = self.optimizer.minimize(self.cross_entropy)

        print(("build model finished: %ds" % (time.time() - start_time)))
        return tf

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, filt_size, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filt_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)

        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)

        # Create & upload a file.
        uploaded = drive.CreateFile({'title': npy_path + '.npy'})
        uploaded.SetContentFile(npy_path + '.npy')
        uploaded.Upload()
        print('Uploaded file with ID {}'.format(uploaded.get('id')))

        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
