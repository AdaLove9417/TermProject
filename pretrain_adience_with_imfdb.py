import tensorflow as tf
import database
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tensorflow-vgg'))
import numpy as np
import levi_and_hassner
import math
import inspect

num_batches = 20000
num_classes = 8
learning_rate = math.e**(-3)

db = database.database()
db.load_set('IMFDB_final/imdb/', 227, 227, 64, 8)

sess = tf.Session()

images = tf.placeholder(tf.float32, [None, 227, 227, 3])
train_mode = tf.placeholder(tf.bool)
pretrain_mode = tf.placeholder(tf.bool)
levi_hassner = levi_and_hassner.LeviHassner()
tf = levi_hassner.build(images, learning_rate, 8, tf.constant(True))

sess.run(tf.global_variables_initializer())

for i in range(1, num_batches):
    [x_batch, y_batch] = db.sample_train()
    [_, accuracy, cross_entropy] = sess.run([levi_hassner.train_op, levi_hassner.accuracy, levi_hassner.cross_entropy],
                                            feed_dict={levi_hassner.x_: x_batch, levi_hassner.y_: y_batch,
                                                       train_mode: True, pretrain_mode: False})
    if i % 500 == 0 or i == 1:
        train_accuracy = []
        for j in range(1, 10):
            train_accuracy.append(sess.run(levi_hassner.accuracy, feed_dict))
        test_accuracy = np.mean(test_accuracy)
        [x_test_batch, y_test_batch] = db.sample_test()
        feed_dict = {levi_hassner.x_: x_test_batch, levi_hassner.y_: y_test_batch}
        for j in range(1, 10):
            test_accuracy.append(sess.run(levi_hassner.accuracy, feed_dict))
        test_accuracy = np.mean(test_accuracy)
        print_str = 'epoch{0} -- train accuracy: {1:.2%} | test accuracy: {2:.2%}'
        print(print_str.format(i, train_accuracy, test_accuracy))
        levi_hassner.save_npy(sess, npy_path='./imfdb_pretrain-epoch-{0}'.format(i))
    else:
        print('epoch{0} -- train batch accuracy: {1:.2%} | loss: {2}'.format(i, accuracy, cross_entropy))