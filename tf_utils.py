import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python import pywrap_tensorflow

import gen_lapnorm

def prelu(inp, name):
    with tf.variable_scope(name):
        i = int(inp.get_shape()[-1])
        alpha = make_var('alpha', shape=(i,))
        output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
    return output

def expanded_conv2d(x, shape, name, strides=[1, 1, 1, 1], activation_func=tf.nn.relu, trainable=True, is_expand=True):
    with tf.variable_scope(name):
        # with tf.variable_scope('expand'):
        if is_expand:
            x = conv2d(x, shape=[1, 1, shape[0], shape[0] * 6],
                       name='expand',
                       use_bn=True,
                       activation_func=activation_func,
                       trainable=trainable)
            shape[0] *= 6
        x = depthwise_conv2d(x, depthwise_filter=[3, 3, shape[0], 1],
                             name='depthwise',
                             activation_func=activation_func,
                             trainable=trainable)
        x = conv2d(x, shape=[1, 1, shape[0], shape[1]],
                   name='project',
                   use_bn=True,
                   strides=strides,
                   activation_func=activation_func,
                   trainable=trainable)
    return x


def separable_conv2d(x, depthwise_filter, pointwise_filter, name, activation_func=tf.nn.relu, strides=[1, 1, 1, 1],
                     trainable=True, padding='SAME', use_bn=True, bn_scale=False):
    with tf.variable_scope(name):
        # print('depthwise_filter:', depthwise_filter)
        depthwise_filter = make_var(name='depthwise_weights', shape=depthwise_filter, trainable=trainable,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
        pointwise_filter = make_var(name='pointwise_weights', shape=pointwise_filter, trainable=trainable,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # x = tf.nn.depthwise_conv2d(x, filter=depthwise_filter,
        #                            strides=strides, padding=padding, name='conv_dw')
        x = tf.nn.separable_conv2d(x, depthwise_filter=depthwise_filter, pointwise_filter=pointwise_filter,
                                   strides=strides, padding=padding)
        # tf.nn.separable_conv2d
        if use_bn:
            # x = batch_normal(x, name='batch_normal', trainable=trainable,)
            x = tf.contrib.layers.batch_norm(x, scale=bn_scale)

        if activation_func is not None:
            x = activation_func(x, name='activate_func')

        # x = conv2d(x, )
    return x

def depthwise_conv2d(x, depthwise_filter, name, activation_func=tf.nn.relu, strides=[1, 1, 1, 1],
               trainable=True, padding='SAME', use_bn=True, bn_scale=False):
    with tf.variable_scope(name):
        # print('depthwise_filter:', depthwise_filter)
        depthwise_filter = make_var(name='depthwise_weights', shape=depthwise_filter, trainable=trainable,
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # pointwise_filter = make_var(name='pointwise_weights', shape=pointwise_filter, trainable=trainable,
        #                             initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.nn.depthwise_conv2d(x, filter=depthwise_filter,
                                   strides=strides, padding=padding, name='conv_dw')
        # tf.nn.separable_conv2d
        if use_bn:
            # x = batch_normal(x, name='batch_normal', trainable=trainable,)
            x = tf.contrib.layers.batch_norm(x, scale=bn_scale)

        if activation_func is not None:
            x = activation_func(x, name='activate_func')

        # x = conv2d(x, )
    return x

def global_pool(input_tensor, pool_op=tf.nn.avg_pool):
    """Applies avg pool to produce 1x1 output.

    NOTE: This function is funcitonally equivalenet to reduce_mean, but it has
    baked in average pool which has better support across hardware.

    Args:
      input_tensor: input tensor
      pool_op: pooling op (avg pool is default)
    Returns:
      a tensor batch_size x 1 x 1 x depth.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size = tf.convert_to_tensor(
            [1, tf.shape(input_tensor)[1],
             tf.shape(input_tensor)[2], 1])
    else:
        kernel_size = [1, shape[1], shape[2], 1]
    output = pool_op(
        input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
    # Recover output shape, for unknown shape.
    output.set_shape([None, 1, 1, None])
    return output

def conv2d(x, shape, name, activation_func=tf.nn.relu, strides=[1, 1, 1, 1],
           trainable=True, padding='SAME', use_bn=True, bn_scale=False):
    with tf.variable_scope(name):
        w = make_var(name='weights', shape=shape, trainable=trainable,
                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.nn.conv2d(x, w, strides=strides, padding=padding, name='conv')
        if use_bn:
            # x = batch_normal(x, name='batch_normal', trainable=trainable,)
            x = tf.contrib.layers.batch_norm(x, scale=bn_scale)
        else:
            b = make_var(name='biases', shape=shape[-1], trainable=trainable)
            x = tf.add(x, b)
        if activation_func is not None:
            x = activation_func(x, name='activate_func')

    return x

def selu(x, name):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def make_var(name,
             shape,
             initializer=tf.contrib.layers.xavier_initializer(),
             dtype='float',
             collections=None,
             trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    # collections = [tf.GraphKeys.GLOBAL_VARIABLES, MODELS_VARIABLES]
    # tf.GraphKeys.GLOBAL_VARIABLES
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=None,
                           collections=collections,
                           trainable=trainable)


def fc(inp, num_out, name, trainable=True):
    # with tf.variable_scope(name):
    input_shape = inp.get_shape()
    if input_shape.ndims == 4:
        # The input is spatial. Vectorize it first.
        dim = 1
        for d in input_shape[1:].as_list():
            dim *= int(d)
        feed_in = tf.reshape(inp, [-1, dim])
    else:
        feed_in, dim = (inp, input_shape[-1].value)
    weights = make_var('weights', shape=[dim, num_out], trainable=trainable,)
    biases = make_var('biases', [num_out], initializer=tf.zeros_initializer(), trainable=trainable)
    fc = tf.nn.xw_plus_b(feed_in, weights, biases, name=name)
    return fc


def batch_normal(x, name, use_bias=False, trainable=True):
    MOVING_AVERAGE_DECAY = 0.9997
    BN_DECAY = MOVING_AVERAGE_DECAY
    BN_EPSILON = 0.001
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if use_bias:
        bias = make_var('bias', params_shape,
                        initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = make_var('beta',
                    params_shape,
                    initializer=tf.zeros_initializer)

    gamma = make_var('gamma',
                     params_shape,
                     initializer=tf.ones_initializer)

    moving_mean = make_var('moving_mean',
                           params_shape,
                           initializer=tf.zeros_initializer,
                           trainable=False)

    moving_variance = make_var('moving_variance',
                               params_shape,
                               initializer=tf.ones_initializer,
                               trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)

    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    # print('trainable:', type(trainable))
    # tf.constant()

    mean, variance = control_flow_ops.cond(
        tf.constant(trainable, dtype=tf.bool), lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON, name=name)
    #x.set_shape(inputs.get_shape()) ??

    return x

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def summary_variables():
    # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables = tf.trainable_variables()
    for v in variables:
        # print('v:', v)
        if 'biases' in v.op.name:
            tf.summary.scalar(v.op.name, v)
        else:
            tf.summary.histogram(v.op.name, v)
        tf.summary.scalar(v.op.name + '/avg', tf.reduce_mean(v))


def visualize(sess, input, img=None, path='./', width=224, height=224):

    if not os.path.exists(path):
        print('path:', path)
        os.makedirs(path)
    graph = sess.graph
    layer_names = [op.name for op in graph.get_operations() if op.type == 'Conv2D']
    # i = 0
    # if img is not None:
    #     pre_img = img.copy()
    for name in layer_names:
        layer_output = graph.get_tensor_by_name(name + ':0')
        print('shape of %s:%s' % (name, str(layer_output.get_shape())))
        if img is None:
            img = np.random.uniform(size=(width, height, 3)) + 100
            img = np.expand_dims(img, axis=0)

        images = img.copy()
        img_arr = gen_lapnorm.render_multiscale(sess, input, layer_output, images)

        gen_lapnorm.savearray(img_arr[0], os.path.join(path, name.replace('/', '_') + '.jpg'))


def inception(x, shapes, name, use_bn=True, trainable=True, activation_func=tf.nn.relu):
    assert len(shapes) == 6
    dims = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        with tf.variable_scope('Branch_0'):
            branch_0 = conv2d(x, name='Conv2d_0a_1x1',
                                       shape=[1, 1, dims, shapes[0]],
                                       strides=[1, 1, 1, 1],
                                       use_bn=use_bn,
                                       trainable=trainable,
                                       activation_func=activation_func)
        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d(x, name='Conv2d_0a_1x1',
                                       shape=[1, 1, dims, shapes[1]],
                                       strides=[1, 1, 1, 1],
                                       use_bn=use_bn,
                                       trainable=trainable,
                                       activation_func=activation_func)
            branch_1 = conv2d(branch_1, name='Conv2d_0b_3x3',
                                       shape=[3, 3, shapes[1], shapes[2]],
                                       strides=[1, 1, 1, 1],
                                       use_bn=use_bn,
                                       trainable=trainable,
                                       activation_func=activation_func)
        with tf.variable_scope('Branch_2'):
            branch_2 = conv2d(x, name='Conv2d_0a_1x1',
                                       shape=[1, 1, dims, shapes[3]],
                                       strides=[1, 1, 1, 1],
                                       use_bn=use_bn,
                                       trainable=trainable,
                                       activation_func=activation_func)
            branch_2 = conv2d(branch_2, name='Conv2d_0b_3x3',
                                       shape=[3, 3, shapes[3], shapes[4]],
                                       strides=[1, 1, 1, 1],
                                       use_bn=use_bn,
                                       trainable=trainable,
                                       activation_func=activation_func)
        with tf.variable_scope('Branch_3'):
            branch_3 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME',
                                      name='MaxPool_0a_3x3')
            branch_3 = conv2d(branch_3, name='Conv2d_0b_1x1',
                                       shape=[1, 1, dims, shapes[5]],
                                       strides=[1, 1, 1, 1],
                                       use_bn=use_bn,
                                       trainable=trainable,
                                       activation_func=activation_func)
    x = tf.concat(
        axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    return x


def get_variables_from_ckpt(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    variables = {}
    for key in var_to_shape_map:
        print("tensor_name: ", key, np.shape(reader.get_tensor(key)))
        variables[key] = reader.get_tensor(key)
