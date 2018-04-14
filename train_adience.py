import tensorflow as tf
import database
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tensorflow-vgg'))
import vgg16
import math


num_batches = 20000
num_classes = 8
learning_rate = 1e-1

db = database.database()
db.load_set('adience/imdb/', 224, 224, 64, 8)

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16.Vgg16()
tf = vgg.build(images, learning_rate, 8, tf.constant(True))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1, num_batches):
    [x_batch, y_batch] = db.sample_train()
    x_batch = x_batch - db.train_avg
    feed_dict = {vgg.x_: x_batch, vgg.y_: y_batch}
    sess.run(vgg.train_op, feed_dict)
    accuracy = sess.run(vgg.accuracy, feed_dict)
    cross_entropy = sess.run(vgg.cross_entropy, feed_dict)
    if i % 1000 == 0 or i == 1:
        [x_test_batch, y_test_batch] = db.sample_test()
        feed_dict = {vgg.x_: x_test_batch, vgg.y_: y_test_batch}
        test_accuracy = sess.run(vgg.accuracy, feed_dict)
        print_str = 'epoch{0} -- train accuracy: {1:.2%} | test accuracy: {2:.2%}'
        print(print_str.format(i, accuracy, test_accuracy))
        vgg.save_npy(sess, npy_path='./vgg-16-epoch-{0}'.format(i))
    else:
        print('epoch{0} -- train accuracy: {1:.2%} -- loss: {2}'.format(i, accuracy, cross_entropy))