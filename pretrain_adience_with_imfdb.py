import tensorflow as tf
import database
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tensorflow-levi_hassner'))
import levi_and_hassner
import math
import inspect

num_batches = 20000
num_classes = 8
learning_rate = 1e-3

db = database.database()
db.load_set('IMFDB_final/imdb/', 227, 227, 64, 8)

images = tf.placeholder(tf.float32, [None, 227, 227, 3])
levi_hassner = levi_hassner.LeviHassener()
tf = levi_hassner.build(images, learning_rate, 8, tf.constant(True))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1, num_batches):
    [x_batch, y_batch] = db.sample_train()
    x_batch = x_batch - db.train_avg
    feed_dict = {levi_hassner.x_: x_batch, levi_hassner.y_: y_batch}
    sess.run(levi_hassner.train_op, feed_dict)
    accuracy = sess.run(levi_hassner.accuracy, feed_dict)
    cross_entropy = sess.run(levi_hassner.cross_entropy, feed_dict)
    learn_rate = sess.run(levi_hassner.learn_rate)
    if i % 1000 == 0 or i == 1:
        [x_test_batch, y_test_batch] = db.sample_test()
        feed_dict = {levi_hassner.x_: x_test_batch, levi_hassner.y_: y_test_batch}
        test_accuracy = sess.run(levi_hassner.accuracy, feed_dict)
        print_str = 'epoch{0} -- train accuracy: {1:.2%} | test accuracy: {2:.2%}'
        print(print_str.format(i, accuracy, test_accuracy))
        levi_hassner.save_npy(sess, npy_path='./imfdb_pretrain-epoch-{0}'.format(i))
    else:
        print('epoch{0} -- train accuracy: {1:.2%} | loss: {2} | lr: {3}'.format(i, accuracy, cross_entropy, learn_rate))