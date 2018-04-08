import tensorflow as tf
import database
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tensorflow-vgg'))
import vgg16
import math


num_batches = 20000
num_classes = 8
learning_rate = math.e**(-3)

db = database.database()
db.load_set('adience/imdb/', 224, 224, 128, 8)

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16.Vgg16()
tf = vgg.build(images, learning_rate, 8, tf.constant(True))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1, num_batches):
    [x_batch, y_batch] = db.sample_train()
    feed_dict = {vgg.x_: x_batch, vgg.y_: y_batch}
    sess.run(vgg.train_op, feed_dict)
    [prob, max_diff] = sess.run([vgg.max_prob, vgg.diff_max], feed_dict)
    print(prob)
    print(max_diff[1])
    #print('epoch{0} -- test accuracy: {1}'.format(i, cross_entropy]))
    vgg.save_npy(sess, npy_path='./vgg-16-epoch-{0}'.format(i))



