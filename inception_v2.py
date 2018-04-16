import tensorflow as tf

import tf_utils

class InceptionV2(object):
    def __init__(self, image, num_classes=1000,
                 use_bn=True,
                 weight_decay=0.0,
                 trainable=True,
                 activation_func=tf.nn.relu):
        # pass
        self.use_bn = use_bn
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.trainable = trainable
        self.activation_func = activation_func

        with tf.variable_scope('InceptionV2', initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(weight_decay)) as scope:
            self.target = self.builder_network(image)

    def builder_network(self, image):
        x = image
        # x = tf_utils.conv2d(x, name='Conv2d_1a_7x7', shape=[7, 7, 3, 8], strides=[1, 2, 2, 1],
        #                     padding='SAME',
        #                     use_bn=self.use_bn,
        #                     trainable=self.trainable,
        #                     activation_func=self.activation_func)
        x = tf_utils.separable_conv2d(x, depthwise_filter=[7, 7, 3, 8], pointwise_filter=[1, 1, 24, 64],
                                      name='Conv2d_1a_7x7',
                                      padding='SAME',
                                      use_bn=False,
                                      trainable=self.trainable,
                                      activation_func=self.activation_func)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='MaxPool_2a_3x3')

        x = tf_utils.conv2d(x, name='Conv2d_2b_1x1', shape=[1, 1, 64, 64], strides=[1, 1, 1, 1],
                            padding='SAME',
                            use_bn=self.use_bn,
                            trainable=self.trainable,
                            activation_func=self.activation_func)

        x = tf_utils.conv2d(x, name='Conv2d_2c_3x3', shape=[3, 3, 64, 192], strides=[1, 1, 1, 1],
                            padding='SAME',
                            use_bn=self.use_bn,
                            trainable=self.trainable,
                            activation_func=self.activation_func)

        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='MaxPool_3a_3x3')

        x = self.inception(x, shapes=[64, 64, 64, 64, 96, 96, 32], name='Mixed_3b')
        x = self.inception(x, shapes=[64, 64, 96, 64, 96, 96, 64], name='Mixed_3c')
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='MaxPool_4a_3x3')
        dims = x.get_shape().as_list()[-1]
        with tf.variable_scope('Mixed_4a'):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf_utils.conv2d(x, name='Conv2d_0a_1x1',
                                           shape=[1, 1, dims, 128],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                branch_0 = tf_utils.conv2d(branch_0, name='Conv2d_1a_3x3',
                                           shape=[3, 3, 128, 160],
                                           strides=[1, 2, 2, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                print('branch_3:', branch_0)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf_utils.conv2d(x, name='Conv2d_0a_1x1',
                                           shape=[1, 1, dims, 64],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                print('branch_3:', branch_1)
                branch_1 = tf_utils.conv2d(branch_1, name='Conv2d_0b_3x3',
                                           shape=[3, 3, 64, 96],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                branch_1 = tf_utils.conv2d(branch_1, name='Conv2d_1a_3x3',
                                           shape=[3, 3, 96, 96],
                                           strides=[1, 2, 2, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='MaxPool_1a_3x3')
                # branch_2 = slim.max_pool2d(
                #     net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
                print('branch_2:', branch_2)

            x = tf.concat(
                axis=3, values=[branch_0, branch_1, branch_2])

        x = self.inception(x, shapes=[224, 64, 96, 96, 128, 128, 128], name='Mixed_4b')
        x = self.inception(x, shapes=[192, 96, 128, 96, 128, 128, 128], name='Mixed_4c')
        x = self.inception(x, shapes=[160, 128, 160, 128, 160, 160, 96], name='Mixed_4d')
        x = self.inception(x, shapes=[96, 128, 192, 160, 192, 192, 96], name='Mixed_4e')
        # x = self.inception(x, shapes=[256, 160, 320, 32, 128, 128], name='Mixed_4f')
        # x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='MaxPool_5a_2x2')
        dims = x.get_shape().as_list()[-1]
        with tf.variable_scope('Mixed_5a'):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf_utils.conv2d(x, name='Conv2d_0a_1x1',
                                           shape=[1, 1, dims, 128],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                branch_0 = tf_utils.conv2d(branch_0, name='Conv2d_1a_3x3',
                                           shape=[3, 3, 128, 192],
                                           strides=[1, 2, 2, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                print('branch_3:', branch_0)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf_utils.conv2d(x, name='Conv2d_0a_1x1',
                                           shape=[1, 1, dims, 192],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                print('branch_3:', branch_1)
                branch_1 = tf_utils.conv2d(branch_1, name='Conv2d_0b_3x3',
                                           shape=[3, 3, 192, 256],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                branch_1 = tf_utils.conv2d(branch_1, name='Conv2d_1a_3x3',
                                           shape=[3, 3, 256, 256],
                                           strides=[1, 2, 2, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='MaxPool_1a_3x3')
                # branch_2 = slim.max_pool2d(
                #     net, [3, 3], stride=2, scope='MaxPool_1a_3x3')

            x = tf.concat(
                axis=3, values=[branch_0, branch_1, branch_2])

        x = self.inception(x, shapes=[352, 192, 320, 160, 224, 224, 128], name='Mixed_5b')
        x = self.inception(x, shapes=[352, 192, 320, 192, 224, 224, 128], name='Mixed_5c')
        print('x:', x)
        with tf.variable_scope('Logits'):
            x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
            x = tf_utils.conv2d(x, shape=[1, 1, 1024, self.num_classes],
                                name='Conv2d_1c_1x1',
                                use_bn=False,
                                padding='SAME')

        # pass
        print('x:', x)
        x = tf.squeeze(x, [1, 2], name='SpatialSqueeze')
        print('x:', x)
        return x

    def inception(self, x, shapes, name):
        assert len(shapes) == 7
        dims = x.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf_utils.conv2d(x, name='Conv2d_0a_1x1',
                                           shape=[1, 1, dims, shapes[0]],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                # print('branch_3:', branch_0)
            with tf.variable_scope('Branch_1'):
                branch_1 = tf_utils.conv2d(x, name='Conv2d_0a_1x1',
                                           shape=[1, 1, dims, shapes[1]],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                # print('branch_3:', branch_1)
                branch_1 = tf_utils.conv2d(branch_1, name='Conv2d_0b_3x3',
                                           shape=[3, 3, shapes[1], shapes[2]],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                # print('branch_3:', branch_1)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf_utils.conv2d(x, name='Conv2d_0a_1x1',
                                           shape=[1, 1, dims, shapes[3]],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                # print('branch_3:', branch_2)
                branch_2 = tf_utils.conv2d(branch_2, name='Conv2d_0b_3x3',
                                           shape=[3, 3, shapes[3], shapes[4]],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                branch_2 = tf_utils.conv2d(branch_2, name='Conv2d_0c_3x3',
                                           shape=[3, 3, shapes[4], shapes[5]],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                # print('branch_3:', branch_2)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1],
                                          strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          name='AvgPool_0a_3x3')
                # print('branch_3:', branch_3)
                branch_3 = tf_utils.conv2d(branch_3, name='Conv2d_0b_1x1',
                                           shape=[1, 1, dims, shapes[6]],
                                           strides=[1, 1, 1, 1],
                                           use_bn=self.use_bn,
                                           trainable=self.trainable,
                                           activation_func=self.activation_func)
                # print('branch_3:', branch_3)
        x = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        return x


if __name__ == '__main__':
    import numpy as np
    # tf_utils.get_variables_from_ckpt('../models/inception_v2.ckpt')
    # exit(0)

    X = tf.placeholder(name='X', shape=[None, None, None, 3], dtype=tf.float32)
    inception_v2 = InceptionV2(X, num_classes=1001,)

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, '../models/inception_v2.ckpt')
        print(sess.run(inception_v2.target, feed_dict={X:np.random.randn(10, 224, 224, 3)}))
        # for v in tf.trainable_variables():
        #     print(v)

        tf_utils.visualize(sess, X, path='../images/inception_v2')
