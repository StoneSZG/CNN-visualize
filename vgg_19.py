import tensorflow as tf

import tf_utils

class vgg_19():
    def __init__(self, image, num_classes=1000,
                 use_bn=False,
                 weight_decay=0.0,
                 trainable=True,
                 activation_func=tf.nn.relu):
        self.use_bn = use_bn
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.trainable = trainable
        self.activation = activation_func
        with tf.variable_scope('vgg_19', initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(weight_decay)) as scope:
            self.target = self.builder_network(image)

    def builder_network(self, image):
        x = image
        with tf.variable_scope('conv1'):
            x = tf_utils.conv2d(x, [3, 3, 3, 64], name='conv1_1',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 64, 64], name='conv1_2',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv2'):
            x = tf_utils.conv2d(x, [3, 3, 64, 128], name='conv2_1',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 128, 128], name='conv2_2',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv3'):
            x = tf_utils.conv2d(x, [3, 3, 128, 256], name='conv3_1',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 256, 256], name='conv3_2',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 256, 256], name='conv3_3',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 256, 256], name='conv3_4',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv4'):
            x = tf_utils.conv2d(x, [3, 3, 256, 512], name='conv4_1',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 512, 512], name='conv4_2',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 512, 512], name='conv4_3',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 512, 512], name='conv4_4',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv5'):
            x = tf_utils.conv2d(x, [3, 3, 512, 512], name='conv5_1',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 512, 512], name='conv5_2',
                                padding='SAME',
                                use_bn=False,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 512, 512], name='conv5_3',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf_utils.conv2d(x, [3, 3, 512, 512], name='conv5_4',
                                padding='SAME',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('fc6'):
            x = tf_utils.conv2d(x, [7, 7, 512, 4096], name='fc6',
                                padding='VALID',
                                use_bn=self.use_bn,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = self.activation(x)

        with tf.name_scope('fc7'):
            x = tf_utils.conv2d(x, [1, 1, 4096, 4096], name='fc7',
                                padding='SAME',
                                use_bn=False,
                                trainable=self.trainable,
                                activation_func=self.activation)
            x = self.activation(x)

        with tf.name_scope('fc8'):
            x = tf_utils.conv2d(x, [1, 1, 4096, self.num_classes], name='fc8',
                                padding='SAME',
                                use_bn=False,
                                trainable=self.trainable,
                                activation_func=self.activation)

        x = tf.squeeze(x, axis=[1, 2], name='squeezed')
        return x

if __name__ == '__main__':
    import numpy as np
    X = tf.placeholder(name='X', shape=[None, None, None, 3], dtype=tf.float32)
    with tf.Session() as sess:
        with tf.name_scope('VGG19'):
            vgg = vgg_19(X)
            vgg.summary_variables()
        saver = tf.train.Saver()
        saver.restore(sess, '../models/vgg_19.ckpt')
        # print('target:', vgg.target)
        print(sess.run(vgg.target, feed_dict={X:np.random.randn(10, 224, 224, 3)}))
        # for v in tf.trainable_variables():
        #     print(v)

        tf_utils.visualize(sess, X, path='../images/vgg19')

