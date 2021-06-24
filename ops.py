import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops
import keras.backend as K
from utils import *

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum=0.98, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.layer_norm(x, center=True, scale=True, scope=self.name)  #tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)#, begin_norm_axis=1, begin_params_axis=-1


def get_fg0(y_true):
    a1 = tf.constant(value=200 / 255.0, shape=[1, 256, 256, 1], dtype=y_true.dtype)

    a1_get_fg = tf.greater(y_true,a1)
    a1_get_fg = tf.cast(a1_get_fg, tf.float32)
    y_true = tf.multiply(a1_get_fg, y_true)
    return y_true
#denoise
def get_fg(y_true):
    a1 = tf.constant(value=200 / 255.0, shape=[16, 256, 256, 1], dtype=y_true.dtype)

    a1_get_fg = tf.greater(y_true,a1)
    a1_get_fg = tf.cast(a1_get_fg, tf.float32)
    y_true = tf.multiply(a1_get_fg, y_true)
    return y_true

#denoise1
def get_fg1(y_true):
    a1 = tf.constant(value=168 / 255.0, shape=[16, 256, 256, 1], dtype=y_true.dtype)

    a1_get_fg = tf.greater(y_true,a1)
    a1_get_fg = tf.cast(a1_get_fg, tf.float32)
    y_true = tf.multiply(a1_get_fg, y_true)
    return y_true


# change gt 50 85 170 same to pred
def delete_noise(y_true,y_pred):
    # print(y_true.shape)
    a1 = tf.constant(value=50 / 255.0,shape=[16,256,256,1],dtype=y_true.dtype)
    a2 = tf.constant(value=85 / 255.0, shape=[16,256,256,1], dtype=y_true.dtype)
    a3 = tf.constant(value=170 / 255.0, shape=[16,256,256,1], dtype=y_true.dtype)

    a1_get_p = tf.equal(y_true,a1)
    a1_get_p = tf.cast(a1_get_p,tf.float32)
    a1_get_t = tf.negative(a1_get_p) + 1.0
    y_true = tf.multiply(a1_get_t,y_true)
    y_pred = tf.multiply(a1_get_t,y_pred)

    a2_get_p = tf.equal(y_true, a2)
    a2_get_p = tf.cast(a2_get_p, tf.float32)
    a2_get_t = tf.negative(a2_get_p) + 1.0
    y_true = tf.multiply(a2_get_t, y_true)
    y_pred = tf.multiply(a2_get_t, y_pred)

    a3_get_p = tf.equal(y_true, a3)
    a3_get_p = tf.cast(a3_get_p, tf.float32)
    a3_get_t = tf.negative(a3_get_p) + 1.0
    y_true = tf.multiply(a3_get_t, y_true)
    y_pred = tf.multiply(a3_get_t, y_pred)


    # y_true[y_true == 50 / 255.0] = 0
    # y_true[y_true == 85 / 255.0] = 0
    # y_true[y_true == 170 / 255.0] = 0
    # y_pred[y_true == 50 / 255.0] = 0
    # y_pred[y_true == 85 / 255.0] = 0
    # y_pred[y_true == 170 / 255.0] = 0
    return y_true,y_pred


def get_area(y_true):
    wl = (65536 * 16) / tf.reduce_sum(y_true)
    return min(tf.Variable(50.0), wl)


def get_area1(y_true,target_num):

    if not tf.equal(tf.reduce_sum(target_num), tf.constant(value=0.0, dtype='float32')):
        average_summ = tf.reduce_sum(y_true) / tf.reduce_sum(target_num)
        wl = (65536) / average_summ
        return min(tf.Variable(50.0),wl)
    else:
        return tf.Variable(50.0)

def get_area2(y_true,target_num,dice):

    if not tf.equal(tf.reduce_sum(target_num), tf.constant(value=0.0, dtype='float32')):
        num_nozero = tf.count_nonzero(target_num)
        wl = (65536 * num_nozero) / (tf.reduce_sum(y_true) * dice)
        return min(tf.Variable(50.0), wl)
    else:
        return tf.Variable(50.0)

def get_area3(y_true,dice):
    wl = (65536 * 16) / (tf.reduce_sum(y_true) * dice)
    return min(tf.Variable(50.0), wl)

def get_area4(y_true,target_num,dice):

    if not tf.equal(tf.reduce_sum(target_num), tf.constant(value=0.0, dtype='float32')):
        num_nozero = tf.count_nonzero(target_num)
        wl = (65536 * num_nozero) / (tf.reduce_sum(y_true))
        return min(tf.Variable(50.0), wl) / dice
    else:
        return tf.Variable(50.0) / dice

def get_area5(y_true,dice):
    wl = (65536 * 16) / tf.reduce_sum(y_true)
    return min(tf.Variable(50.0), wl) / dice


def get_area6(y_true,target_num):
    if not tf.equal(tf.reduce_sum(target_num), tf.constant(value=0.0, dtype='float32')):
        num_nozero = tf.count_nonzero(target_num)
        wl = (65536 * num_nozero) / (tf.reduce_sum(y_true))
        return min(tf.Variable(50.0), wl)
    else:
        return tf.Variable(50.0)

def get_area7(y_true,target_num):
    if not tf.equal(tf.reduce_sum(target_num), tf.constant(value=0.0, dtype='float32')):
        num_nozero = tf.count_nonzero(target_num)
        wl = (65536 * 16) / (tf.reduce_sum(y_true))
        return min(tf.Variable(50.0 + 2.0 * num_nozero), wl)
    else:
        return tf.Variable(50.0)




# use area value of fg to get beta
def get_beta(y_true):   #(0.5 - 1)
    summ = tf.reduce_sum(y_true)
    beta = 1.05 - 2.5 * summ / (65536 * 16)
    return beta

def get_beta1(y_true):    #(1.0 - 1.5)
    summ = tf.reduce_sum(y_true)
    beta = 1.55 - 2.5 * summ / (65536 * 16)
    return beta

def get_beta2(y_true):
    summ = tf.reduce_sum(y_true)
    beta = 5.6 - 20.0 * summ / (65536 * 16)
    return beta

def get_beta3(y_true):
    summ = tf.reduce_sum(y_true)
    beta = 0.33 + 21.0 * summ / (65536 * 16)
    return beta


def sigmoid_focal_crossentropy_new1(y_true, y_pred, alpha=0.25, gamma=2.0,beta=1.0, from_logits=True):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
    # ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha * beta + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
def sigmoid_focal_crossentropy_new(y_true, y_pred, alpha=0.25, gamma=2.0,beta=1.0, from_logits=False):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha * beta + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

# alpha: balancing factor ,gamma modulating factor
def sigmoid_focal_crossentropy( y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true,dtype=y_pred.dtype)
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

# XY / GT
def XY_GT(y_true,y_pred,smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + smooth)

#dice loss
def dice_coef(y_true,y_pred,smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth ) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def atrous5_conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.atrous_conv2d(input_, w, rate=2, padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases) #tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
def atrous7_conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.atrous_conv2d(input_, w, rate=3, padding='SAME')
        # tf.nn.atrous_conv2d_transpose
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases) #tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev,seed=38))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases) #tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev,seed=334))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev,seed=786))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start,seed=3778))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def max_average_pool(x):
    return tf.nn.max_pool(x, ksize=[1, x.get_shape()[-3], x.get_shape()[-2], 1],
                          strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def conv2d_c(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape, name='weight'):
        # 截尾正态分布,stddev是正态分布的标准偏差
    with tf.variable_scope(name):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

def bias_variable(shape, name='bias'):

    with tf.variable_scope(name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)