import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor
from ops import *
from toimg import toimg
from ops_ada import cont_Blk
class UNetwork():
    def __init__(self, input_shape, output_shape, drop_out=False):
        if input_shape != output_shape: self.padding = "same"
        else: self.padding = "same"
        self.drop_out = drop_out
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.logits = self.model(input_shape)
        self.labels = tf.placeholder(tf.float32, [None]+output_shape)
        self.low_loss =  tf.reduce_mean(tf.square(tf.image.resize_nearest_neighbor(self.labels,(16,16)) - self.I16)) + tf.reduce_mean(tf.square(tf.image.resize_nearest_neighbor(self.labels,(32,32)) - self.I32)) + tf.reduce_mean(tf.square(tf.image.resize_nearest_neighbor(self.labels,(64,64)) - self.I64))+ tf.reduce_mean(tf.square(self.labels- self.I128))
        self.loss = tf.reduce_mean(tf.square(self.labels - self.logits)) + self.regularization_loss() +self.low_loss
        self.training = self.optimizer.minimize(self.loss)


    def model(self, input_shape):
        self.input1= tf.placeholder(tf.float32, [None]+input_shape)
        self.input2 = tf.placeholder(tf.float32, [None] + input_shape)
        self.input3 = tf.placeholder(tf.float32, [None] + input_shape)
        self.input4 = tf.placeholder(tf.float32, [None] + input_shape)
        self.input5 = tf.placeholder(tf.float32, [None] + input_shape)
        self.input6 = tf.placeholder(tf.float32, [None] + input_shape)
        self.input7 = tf.placeholder(tf.float32, [None] + input_shape)
        self.bmode = tf.placeholder(tf.float32, [None] + [128,128,1])
        self.cond_ph = tf.placeholder(tf.float32, [1, 4])
        with tf.variable_scope('encoder', reuse=False):
            channel=16
            xA = self.input1
            xA = self.conv_encode12(xA, channel,  pad=1, use_bias=True, sn=True, scope='conv')
            xB = self.input2
            xB = self.conv_encode12(xB, channel, pad=1, use_bias=True, sn=True, scope='convB')
            x=tf.concat([xA, xB], axis=-1)
            xC = self.input3
            xC = self.conv_encode12(xC, channel,   pad=1, use_bias=True, sn=True, scope='convC')
            x=tf.concat([x, xC], axis=-1)
            xD = self.input4
            xD = self.conv_encode12(xD, channel,   pad=1, use_bias=True, sn=True, scope='convD')
            x=tf.concat([x, xD], axis=-1)
            xE = self.input5
            xE = self.conv_encode12(xE, channel,  pad=1, use_bias=True, sn=True, scope='convE')
            x=tf.concat([x, xE], axis=-1)
            xF = self.input6
            xF = self.conv_encode12(xF, channel,  pad=1, use_bias=True, sn=True, scope='convF')
            x=tf.concat([x, xF], axis=-1)
            xG = self.input7
            xG = self.conv_encode12(xG, channel,  pad=1, use_bias=True, sn=True, scope='convG')
            x=tf.concat([x, xG], axis=-1)
            for i in range(3):
                x = tf.keras.activations.relu(x)
                x = self.conv_encode12(x, channel * 2,   pad=1, use_bias=True, sn=True, scope='conv_' + str(i))
                x = self.conditional_instance_norm(x, scope_bn=str(i), y1=self.cond_ph)
                x = tf.keras.layers.Dropout(rate=0.5)(x)
                channel = channel * 2
            for i in range(2):
                x = tf.keras.activations.relu(x)
                x = self.conv_encode(x, channel * 2,   pad=1, stride =2, use_bias=True, sn=True, scope='conv_2' + str(i))
                x = self.conditional_instance_norm(x, scope_bn='2_'+str(i), y1=self.cond_ph)
                x = tf.keras.layers.Dropout(rate=0.5)(x)
                print(x)
                channel = channel*2
            x = tf.keras.activations.relu(x)
        x = tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]])
        x =  tf.layers.conv2d(inputs=x, filters=channel,
                                     kernel_size=3, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=[2,3])
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        ext = tf.keras.activations.relu(x)
        bm_128, bm_64, bm_32, bm_16 =cont_Blk(self.bmode)
        # Contracting Path
        x = self.HR_INV_module_bm(ext,bm_128, bm_64, bm_32, bm_16)
        return x

    def resblk(self,input, channel = 64,scope='rs'):
        with tf.variable_scope(scope, reuse=False):
            x = self.conv(input, channel)
            x = self.conv_noact(x, channel)
            x = x+input
            x = tf.keras.activations.relu(x)
        return x

    def sinblk(self, input, channel = 64,scope='sin'):
        with tf.variable_scope(scope, reuse=False):
            x = self.resblk(input, channel,scope = 'rs1')
            x = self.resblk(x, channel,scope = 'rs2')
            x = self.resblk(x, channel,scope = 'rs3')
            x = self.resblk(x, channel,scope = 'rs4')
        return x

    def sinblk2(self, input, channel = 64,scope='sn'):
        with tf.variable_scope(scope, reuse=False):

            x = self.resblk(input, channel,scope = 'rs1')
            x = self.resblk(x, channel,scope = 'rs2')
        return x


    def conditional_instance_norm(self,x, scope_bn, y1):
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        beta = tf.get_variable(name=scope_bn + 'beta',
                               shape=[y1.get_shape().as_list()[-1], x.get_shape().as_list()[-1]],
                               initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
        gamma = tf.get_variable(name=scope_bn + 'gamma',
                                shape=[y1.get_shape().as_list()[-1], x.get_shape().as_list()[-1]],
                                initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        beta1 = tf.matmul(y1, beta)
        gamma1 = tf.matmul(y1, gamma)

        x = tf.nn.batch_normalization(x, mean, var, beta1, gamma1, 1e-10)
        return x


    def HR_INV_module_bm(self, input,bm_128, bm_64, bm_32, bm_16):
        x = self.conv(input, 512,scope='cn1')
        ## first
        x = spade(bm_16, x, 512, use_bias=True, sn=False, scope='spade16')

        x_16_1 = self.sinblk(x, 512,scope='sin1')

        ## second
        x_16_2 = x_16_1
        x_32_2 = self.up(x_16_1, 256)
        x_32_2 = spade(bm_32, x_32_2, 256, use_bias=True, sn=False, scope='spade32')

        x_16_2 = self.sinblk(x_16_2, 512,scope='sin2')
        x_32_2 = self.sinblk(x_32_2, 256,scope='sin3')

        ## third
        x_16_3 = tf.concat([x_16_2, self.dn(x_32_2, 64)], -1)
        x_16_3 = self.conv(x_16_3, 512,scope='cn2')

        x_32_3 = tf.concat([x_32_2, self.up(x_16_2, 128)], -1)
        x_32_3 = self.conv(x_32_3, 256,scope='cn3')

        x_64_3 = tf.concat([self.up(x_32_3, 128), self.up(self.up(x_16_2, 256), 128)], -1)
        x_64_3 = self.conv(x_64_3, 128,scope='cn4')
        x_64_3 = spade(bm_64, x_64_3, 128, use_bias=True, sn=False, scope='spade64')

        x_16_3 = self.sinblk(x_16_3, 512,scope='sin5')
        x_32_3 = self.sinblk(x_32_3, 256,scope='sin6')
        x_64_3 = self.sinblk(x_64_3, 128,scope='sin7')


        # fourth

        x_16_4 = tf.concat([self.dn(x_32_3, 512), self.dn(self.dn(x_64_3, 256), 512)], -1)
        x_16_4 = tf.concat([x_16_3, x_16_4], -1)
        x_16_4 = self.conv(x_16_4, 512,scope='cn5')

        x_32_4 = tf.concat([self.up(x_16_3, 256), self.dn(x_64_3, 256)], -1)
        x_32_4 = tf.concat([x_32_4, x_32_3], -1)
        x_32_4 = self.conv(x_32_4, 256,scope='cn6')

        x_64_4 = tf.concat([self.up(x_32_3, 128), self.up(self.up(x_16_3, 256), 128)], -1)
        x_64_4 = tf.concat([x_64_4, x_64_3], -1)
        x_64_4 = self.conv(x_64_4, 128,scope='cn7')

        x_128_4 = tf.concat([self.up(self.up(x_32_3, 128), 64), self.up(self.up(self.up(x_16_3, 256), 128), 64)], -1)
        x_128_4 = tf.concat([x_128_4, self.up(x_64_3, 64)], -1)
        x_128_4 = self.conv(x_128_4, 64,scope='cn8')
        x_128_4 = spade(bm_128, x_128_4, 64, use_bias=True, sn=False, scope='spade128')

        x_128_4 = self.sinblk(x_128_4, 64,scope='sin8')
        x_64_4 = self.sinblk(x_64_4, 128,scope='sin9')
        x_32_4 = self.sinblk(x_32_4, 256,scope='sin10')
        x_16_4 = self.sinblk(x_16_4, 512,scope='sin11')



        self.I16 = toimg(x_16_4, 16)
        self.I32 = toimg(x_32_4, 32)
        self.I64 = toimg(x_64_4, 64)
        self.I128 = toimg(x_128_4, 128)

        x_128_5 = tf.concat([self.up(self.up(x_32_4, 128), 64), self.up(self.up(self.up(x_16_4, 256), 128), 64)], -1)
        x_128_5 = tf.concat([x_128_5, self.up(x_64_4, 64)], -1)
        x_128_5 = tf.concat([x_128_5, x_128_4], -1)
        x_128_5 = self.conv(x_128_5, 64,scope='cn10')
        x_128_5 = self.sinblk(x_128_5, 64)
        x = self.conv(x_128_5, 1, kernel_size=1, activation=tf.keras.activations.sigmoid,scope='cn11')
        return x



    def dn(self, input, channel):
        x = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=[2, 2],
                                     padding=self.padding, activation=None)(input)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.activations.relu(x)
        return x

    def up(self, input, channel):
        x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(input)
        x = tf.keras.layers.Conv2D(filters=channel, kernel_size=1, strides=[1, 1],
                                    activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.activations.relu(x)
        return x

    def conv(self, input, num_outputs, kernel_size=3, stride=1, activation=tf.keras.activations.relu, scope='conv_0'):
        with tf.variable_scope(scope, reuse=False):

            w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, input.get_shape()[-1], num_outputs],
                                initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=input, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='SAME')
            bias = tf.get_variable("bias", [num_outputs], initializer=tf.constant_initializer(0.0))
            tmp = tf.nn.bias_add(x, bias)

            # tmp = tf.keras.layers.Conv2D(filters = num_outputs, kernel_size = kernel_size, strides = [stride, stride], padding = self.padding, activation = None)(input)

            tmp = tf.keras.layers.BatchNormalization()(tmp)
            tmp = activation(tmp)
        if self.drop_out == True: return tf.keras.layers.Dropout(rate=0.5)(tmp)
        else: return tmp

    def conv_noact(self, input, num_outputs, kernel_size=3, stride=1, scope='conv_nc_0'):
        with tf.variable_scope(scope, reuse=False):
            w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, input.get_shape()[-1], num_outputs],
                                initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=input, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='SAME')
            bias = tf.get_variable("bias", [num_outputs], initializer=tf.constant_initializer(0.0))
            tmp = tf.nn.bias_add(x, bias)
            # tmp = tf.keras.layers.Conv2D(filters = num_outputs, kernel_size = kernel_size, strides = [stride, stride], padding = self.padding, activation = None)(input)
            tmp = tf.keras.layers.BatchNormalization()(tmp)
        if self.drop_out == True: return tf.keras.layers.Dropout(rate=0.5)(tmp)
        else: return tmp

    def max_pool(self, input, pool_size=2, stride=2):
        return tf.keras.layers.MaxPool2D(pool_size=(pool_size, pool_size), strides=stride)(input)


    def copy_and_crop(self, source, target):
        source_h = int(source.get_shape().as_list()[1])
        source_w = int(source.get_shape().as_list()[2])
        target_h = int(target.get_shape().as_list()[1])
        target_w = int(target.get_shape().as_list()[2])
        offset_h = int((source_h - target_h)/2)
        offset_w = int((source_w - target_w)/2)
        #crop = tf.image.crop_to_bounding_box(source, offset_h, offset_w, target_h, target_w)
        copy = tf.concat([source, target], -1)
        return copy

    def conv_encode(self,x, channels, kernel=3, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
        factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
        weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
        # tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

        weight_regularizer = None
        weight_regularizer_fully = None
        with tf.variable_scope(scope):
            if pad > 0:
                h = x.get_shape().as_list()[1]
                if h % 2 == 0:
                    pad = pad * 2
                else:
                    pad = max(kernel - (h % stride), 0)

                pad_top = pad // 2
                pad_bottom = pad - pad_top
                pad_left = pad // 2
                pad_right = pad - pad_left

                if pad_type == 'zero':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                if pad_type == 'reflect':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=weight_init,
                                    regularizer=weight_regularizer)
                x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                                 strides=[1, stride, stride, 1], padding='VALID')
                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                    x = tf.nn.bias_add(x, bias)

            else:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=stride, use_bias=use_bias)

            return x

    def conv_encode12(self,x, channels, kernel=3, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
        factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
        weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
        # tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

        weight_regularizer = None
        weight_regularizer_fully = None
        with tf.variable_scope(scope):
            if pad > 0:
                h = x.get_shape().as_list()[1]
                if h % 2 == 0:
                    pad = pad * 2
                else:
                    pad = max(kernel - (h % 2), 0)

                pad_top = pad // 2
                pad_bottom = pad - pad_top
                pad_left = pad // 2
                pad_right = pad - pad_left

                if pad_type == 'zero':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                if pad_type == 'reflect':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=weight_init,
                                    regularizer=weight_regularizer)
                x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                                 strides=[1, 1, 2, 1], padding='VALID')
                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                    x = tf.nn.bias_add(x, bias)

            else:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=[1,2], use_bias=use_bias)

            return x

    def fully_connected(self,x, units, use_bias=True, sn=False, scope='linear'):
        factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
        weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
        # tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

        weight_regularizer = None
        weight_regularizer_fully = None
        with tf.variable_scope(scope):
            x = flatten(x)
            shape = x.get_shape().as_list()
            channels = shape[-1]

            if sn:
                w = tf.get_variable("kernel", [channels, units], tf.float32,
                                    initializer=weight_init, regularizer=weight_regularizer_fully)
                if use_bias:
                    bias = tf.get_variable("bias", [units],
                                           initializer=tf.constant_initializer(0.0))

                    x = tf.matmul(x, spectral_norm(w)) + bias
                else:
                    x = tf.matmul(x, spectral_norm(w))

            else:
                x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                    kernel_regularizer=weight_regularizer_fully,
                                    use_bias=use_bias)

            return x

    def regularization_loss(self):
        """
        If you want to use "Regularization"
        g_loss += regularization_loss('generator')
        d_loss += regularization_loss('discriminator')
        """
        collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss = []
        for item in collection_regularization:
            loss.append(item)

        return tf.reduce_sum(loss)