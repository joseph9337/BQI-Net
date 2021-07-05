import tensorflow as tf
import numpy as np
from collections import OrderedDict

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]
def cont_Blk( b_mode):
    with tf.variable_scope('const_block'):
        channel = 64
        x = conv(b_mode, channel, kernel=3, stride=1, scope='conv_s1_1')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s1_1')
        bm_128=x
        x = conv(x, channel, kernel=3, stride=1, scope='conv_s1_2')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s1_2')
        x = conv(x, channel, kernel=3, stride=1, scope='conv_s1_3')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s1_3')

        x = conv(x, channel, kernel=3, stride=2, scope='conv_s1_s')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s1_s')
        bm_64=x

        x = conv(x, channel, kernel=3, stride=1, scope='conv_s2_2')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s2_2')
        x = conv(x, channel, kernel=3, stride=1, scope='conv_s2_3')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s2_3')
        x = conv(x, channel, kernel=3, stride=2, scope='conv_s2_s')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s2_s')
        bm_32 = x

        x = conv(x, channel, kernel=3, stride=1, scope='conv_s3_2')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s3_2')
        x = conv(x, channel, kernel=3, stride=1, scope='conv_s3_3')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s3_3')
        x = conv(x, channel, kernel=3, stride=2, scope='conv_s3_s')
        x = lrelu(x, 0.2)
        x = instance_norm(x, scope='inst_s3_s')
        bm_16 = x



    return bm_128, bm_64, bm_32, bm_16

def get_weight(weight_shape, gain, lrmul):
    fan_in = np.prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)  # He init

    # equalized learning rate
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul

    # create variable.
    weight = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32,
                             initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
    return weight

def conv(x, channels, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        weight_shape = [kernel, kernel, x.get_shape().as_list()[-1], channels]

        weight = get_weight(weight_shape, gain, lrmul)

        if sn :
            weight = spectral_norm(weight)

        x = tf.nn.conv2d(input=x, filter=weight, strides=[1, stride, stride, 1], padding='SAME')

        return x



def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)




def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def instance_norm(x, scope='instnorm',epsilon=1e-8):
    with tf.variable_scope(scope):
        x = x - tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)

    return x

def regularization_loss(scope_name) :
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization :
        if scope_name in item.name :
            loss.append(item)

    return tf.reduce_sum(loss)
