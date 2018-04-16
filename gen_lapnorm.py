import scipy
import numpy as np
import tensorflow as tf

k = np.float32([1, 4, 5, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)

#图像拆分为低频、高频成分
def lap_split(img):
    # print('lap_split, img shape:', np.shape(img))
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')

        lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])

        hi = img - lo2

    return lo, hi


#将图像img分成n层拉普拉斯金字塔
def lap_split_n(img, n):
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std, eps)


#拉普拉斯金字塔标准化
def lap_normalize(img, scale_n=4):
    # img = tf.expand_dims(img, 0)
    tlevels = lap_split_n(img, scale_n)
    #每一层做一次normalize_std
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0, :, :, :]

def calc_grad_tiled(sess, input, img, t_grad, tile_size=512, lap_n=4):
    #每次只对tile_size * tile_size大小的图像计算梯度，避免内存问题
    sz = tile_size
    h, w = img.shape[:2]
    #img_shift:先在行上整体移动，再在列上整体移动
    #防止在tile边缘产生边缘效应
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    lap_grad = lap_normalize(t_grad, lap_n)
    #y, x是开始位置的像素
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            #每次对sub计算梯度。sub的大小是tile_size * tile_size
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(lap_grad, {input:sub})
            grad[y:y + sz, x: x + sz] = g

    #使用np.roll移动回去
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def resize_ratio(img, ratio):
    # print('img shape:', np.shape(img))
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img[0], ratio))
    img = img / 255 * (max - min) + min
    # print('img shape:', np.shape(img))
    img = np.expand_dims(img, axis=0)
    return img

def render_multiscale(sess, t_input, t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
    #t_score是优化目标。它是t_obj的平均值
    t_score = tf.reduce_mean(t_obj)
    #计算t_score对t_input的梯度
    t_grad = tf.gradients(t_score, t_input)[0]

    #创建新图
    img = img0.copy()
    for octave in range(octave_n):
        if octave != 1.0:
            img = resize_ratio(img, octave_scale)
        for i in range(iter_n):
            #调用calc_grad_tile计算任意大小图像的梯度
            g = calc_grad_tiled(sess, t_input, img, t_grad, lap_n=lap_n)
            # g = lap_normalize(g, lap_n)
            g /= g.std() + 1e-8
            img += g * step

    return img

def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img save: %s' % img_name)



