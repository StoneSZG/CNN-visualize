import tensorflow as tf

import tf_utils

class MobilenetV2(object):
    def __init__(self, image, num_classes=1000,
                 depth_multiplier=1.0,
                 min_depth=8,
                 weight_decay=0.0,
                 is_training=True,
                 activation_func=tf.nn.relu):
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.trainable = is_training
        self.activation_func = activation_func
        self.depth_multiplier = depth_multiplier
        self.min_depth = min_depth
        self.conv_defs = [{'kernel':[3, 3], 'stride':2, 'depth':32, 'type':'conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':64, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':2, 'depth':128, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':128, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':2, 'depth':256, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':256, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':2, 'depth':512, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':512, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':512, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':512, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':512, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':512, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':2, 'depth':1024, 'type':'separable_conv2d'},
                          {'kernel':[3, 3], 'stride':1, 'depth':1024, 'type':'separable_conv2d'}]
        with tf.variable_scope('MobilenetV2', initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(weight_decay)) as scope:
            self.target = self.builder_network(image)

    def builder_network(self, image):
        x = image
        x = tf_utils.conv2d(x, shape=[3, 3, 3, 32], strides=[1, 2, 2, 1], name='Conv', padding='SAME')
        x = tf_utils.expanded_conv2d(x, shape=[32, 16], name='expanded_conv', is_expand=False)
        x = tf_utils.expanded_conv2d(x, shape=[16, 24], strides=[1, 2, 2, 1], name='expanded_conv_1')
        x = tf_utils.expanded_conv2d(x, shape=[24, 24], name='expanded_conv_2')
        x = tf_utils.expanded_conv2d(x, shape=[24, 32], strides=[1, 2, 2, 1], name='expanded_conv_3')
        x = tf_utils.expanded_conv2d(x, shape=[32, 32], name='expanded_conv_4')
        x = tf_utils.expanded_conv2d(x, shape=[32, 32], name='expanded_conv_5')
        x = tf_utils.expanded_conv2d(x, shape=[32, 64], strides=[1, 2, 2, 1], name='expanded_conv_6')
        x = tf_utils.expanded_conv2d(x, shape=[64, 64], name='expanded_conv_7')
        x = tf_utils.expanded_conv2d(x, shape=[64, 64], name='expanded_conv_8')
        x = tf_utils.expanded_conv2d(x, shape=[64, 64], name='expanded_conv_9')
        x = tf_utils.expanded_conv2d(x, shape=[64, 96], name='expanded_conv_10')
        x = tf_utils.expanded_conv2d(x, shape=[96, 96], name='expanded_conv_11')
        x = tf_utils.expanded_conv2d(x, shape=[96, 96], name='expanded_conv_12')
        x = tf_utils.expanded_conv2d(x, shape=[96, 160], strides=[1, 2, 2, 1], name='expanded_conv_13')
        x = tf_utils.expanded_conv2d(x, shape=[160, 160], name='expanded_conv_14')
        x = tf_utils.expanded_conv2d(x, shape=[160, 160], name='expanded_conv_15')
        x = tf_utils.expanded_conv2d(x, shape=[160, 320], name='expanded_conv_16')
        x = tf_utils.conv2d(x, shape=[1, 1, 320, 1280], name='Conv_1', padding='SAME')


        with tf.variable_scope('Logits'):
            x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
            x = tf_utils.conv2d(x, shape=[1, 1, 1280, self.num_classes],
                                name='Conv2d_1c_1x1',
                                use_bn=False,
                                padding='SAME')


        x = tf.squeeze(x, [1, 2], name='SpatialSqueeze')

        return x

if __name__ == '__main__':
    import numpy as np

    X = tf.placeholder(name='X', shape=[None, None, None, 3], dtype=tf.float32)
    mobilenet_v2 = MobilenetV2(X, num_classes=1001, depth_multiplier=1.0)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, '../models/mobilenet_v2_1.0_224.ckpt')

        print(sess.run(mobilenet_v2.target, feed_dict={X:np.random.randn(10, 224, 224, 3)}))
        # for v in tf.trainable_variables():
        #     print(v)

        tf_utils.visualize(sess, X, path='../images/mobilenet_v2')