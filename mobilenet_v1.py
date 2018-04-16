import tensorflow as tf

import tf_utils

class MobilenetV1(object):
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
        with tf.variable_scope('MobilenetV1', initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(weight_decay)) as scope:
            self.target = self.builder_network(image)

    def builder_network(self, image):
        pre_strides = image.get_shape().as_list()[-1]
        x = image
        depth = lambda d: max(int(d * self.depth_multiplier), self.min_depth)
        with tf.name_scope('conv1'):
            conv = self.conv_defs[0]
            x = self.conv2d(x, [*conv['kernel'], pre_strides, depth(conv['depth'])],
                                strides=[1, conv['stride'], conv['stride'], 1],
                                name='Conv2d_0',
                                activation_func=self.activation_func,
                                padding='SAME')
            pre_strides = depth(conv['depth'])

        for i, conv in enumerate(self.conv_defs[1:]):
            if conv['type'] == 'conv2d':
                x = self.conv2d(x, [*conv['kernel'], pre_strides, depth(conv['depth'])],
                                    strides=[1, conv['stride'], conv['stride'], 1],
                                    name='Conv2d_' + str(i + 1) + '_pointwise',
                                    activation_func=self.activation_func,
                                    padding='SAME')
            elif conv['type'] == 'separable_conv2d':
                x = self.depthwise_conv2d(x, depthwise_filter=[*conv['kernel'], pre_strides, 1],
                                                    strides=[1, 1, 1, 1], padding='SAME',
                                                    name='Conv2d_' + str(i + 1) + '_depthwise',
                                                    activation_func=self.activation_func,
                                                    use_bn=False)

                x = self.conv2d(x, [1, 1, pre_strides, depth(conv['depth'])],
                                    strides=[1, conv['stride'], conv['stride'], 1],
                                    name='Conv2d_' + str(i + 1) + '_pointwise',
                                    activation_func=self.activation_func,
                                    padding='SAME')

            pre_strides = depth(conv['depth'])


        x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 2, 2, 1], padding='VALID',)

        with tf.variable_scope('Logits'):
            # kernel_h, kernel_w = self.weights[-1]
            x = tf_utils.conv2d(x, [1, 1, pre_strides, self.num_classes], name='Conv2d_1c_1x1', padding='SAME',
                                use_bn=False,
                                activation_func=None)

            x = tf.squeeze(x, [1, 2], name='SpatialSqueeze')

        return x

    def depthwise_conv2d(self, x, depthwise_filter, name, activation_func=tf.nn.relu, strides=[1, 1, 1, 1],
                         trainable=True, padding='SAME', use_bn=True):
        with tf.variable_scope(name):
            depthwise_filter = tf_utils.make_var(name='depthwise_weights', shape=depthwise_filter, trainable=trainable,
                                        initializer=tf.contrib.layers.xavier_initializer_conv2d())
            x = tf.nn.depthwise_conv2d(x, filter=depthwise_filter,
                                       strides=strides, padding=padding, name='conv_dw')
            if use_bn:
                x = tf.contrib.layers.batch_norm(x)

            if activation_func is not None:
                x = activation_func(x, name='activate_func')

        return x

    def conv2d(self, x, shape, name, activation_func=tf.nn.relu, strides=[1, 1, 1, 1],
           trainable=True, padding='SAME', use_bn=True):
        with tf.variable_scope(name):
            w = tf_utils.make_var(name='weights', shape=shape, trainable=trainable,
                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            x = tf.nn.conv2d(x, w, strides=strides, padding=padding, name='conv')
            if use_bn:
                # x = batch_normal(x, name='batch_normal', trainable=trainable,)
                x = tf.contrib.layers.batch_norm(x)
            else:
                b = tf_utils.make_var(name='biases', shape=shape[-1], trainable=trainable)
                x = tf.add(x, b)
            if activation_func is not None:
                x = activation_func(x, name='activate_func')

        return x

if __name__ == '__main__':
    import numpy as np

    X = tf.placeholder(name='X', shape=[None, None, None, 3], dtype=tf.float32)
    mobilenet_v1 = MobilenetV1(X, num_classes=1001, depth_multiplier=1.0)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, '../models/mobilenet_v1_1.0_224.ckpt')

        print(sess.run(mobilenet_v1.target, feed_dict={X:np.random.randn(10, 224, 224, 3)}))
        # for v in tf.trainable_variables():
        #     print(v)

        tf_utils.visualize(sess, X, path='../images/mobilenet_v1')


















