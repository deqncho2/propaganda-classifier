import tensorflow as tf 
import numpy as np

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,  'propaganda_autoencoder.ckpt')

    res, loss_test = sess.run([self.decoder_op, self.loss], feed_dict={self.X: datac})

    print(loss_test)
    return res