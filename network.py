import tensorflow as tf
import tensorflow.contrib.layers as slim
import layers

ACTIVATION = layers.Swish
DATA_FORMAT = 'NCHW'

class GeneratorConfig:
    def __init__(self):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.in_channels = 2
        self.out_channels = 2
        # model parameters
        self.activation = ACTIVATION
        self.normalization = 'Instance'
        # train parameters
        self.random_seed = 0
        self.var_ema = 0.999
        self.weight_decay = 1e-6

class Generator(GeneratorConfig):
    def __init__(self, name='Generator', config=None):
        super().__init__()
        self.name = name
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        # create a moving average object for trainable variables
        if self.var_ema > 0:
            self.ema = tf.train.ExponentialMovingAverage(self.var_ema)

    def ResBlock(self, last, channels, kernel=[1, 3], stride=[1, 1], biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        # residual connection
        last = layers.SEUnit(last, channels, format, collections)
        last += skip
        return last

    def EBlock(self, last, channels, resblocks=1,
        kernel=[1, 4], stride=[1, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        # pre-activation
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        # ResBlocks
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, channels, format=format,
                    activation=activation, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        # squeeze and excitation
        # last = layers.SEUnit(last, channels, format, collections)
        return last

    def DBlock(self, last, channels, resblocks=1,
        kernel=[1, 3], stride=[1, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        # shape = last.shape.as_list()
        # in_channels = shape[-3 if format == 'NCHW' else -1]
        # pre-activation
        if activation: last = activation(last)
        # upsample
        with tf.variable_scope('Upsample'):
            upsize = tf.shape(last)
            upsize = upsize[-2:] if format == 'NCHW' else upsize[-3:-1]
            upsize = upsize * stride[0:2]
            if format == 'NCHW':
                last = tf.transpose(last, (0, 2, 3, 1))
            last = tf.image.resize_nearest_neighbor(last, upsize)
            if format == 'NCHW':
                last = tf.transpose(last, (0, 3, 1, 2))
        # convolution
        last = slim.conv2d(last, channels, kernel, [1, 1], 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        # ResBlocks
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, channels, format=format,
                    activation=activation, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        # squeeze and excitation
        # last = layers.SEUnit(last, channels, format, collections)
        return last

    def __call__(self, last, reuse=None):
        format = self.data_format
        kernel1 = [1, 8]
        stride1 = [1, 2]
        kernel2 = [1, 3]
        stride2 = [1, 2]
        # function objects
        activation = self.activation
        if self.normalization == 'Batch':
            normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=True,
                is_training=self.training, data_format=format, renorm=False)
        elif self.normalization == 'Instance':
            normalizer = lambda x: slim.instance_norm(x, center=True, scale=True, data_format=format)
        elif self.normalization == 'Group':
            normalizer = lambda x: (slim.group_norm(x, x.shape.as_list()[-3] // 16, -3, (-2, -1))
                if format == 'NCHW' else slim.group_norm(x, x.shape.as_list()[-1] // 16, -1, (-3, -2)))
        else:
            normalizer = None
        regularizer = slim.l2_regularizer(self.weight_decay) if self.weight_decay else None
        skip_connection = lambda x, y: x + y
        # skip_connection = lambda x, y: tf.concat([x, y], -3 if format == 'NCHW' else -1)
        # model scope
        with tf.variable_scope(self.name, reuse=reuse):
            # states
            self.training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            skips = []
            # encoder
            with tf.variable_scope('InBlock'):
                last = self.EBlock(last, 16, 0, [1, 8], [1, 1],
                    format, None, None, regularizer)
            with tf.variable_scope('EBlock_0'):
                skips.append(last)
                last = self.EBlock(last, 32, 0, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_1'):
                skips.append(last)
                last = self.EBlock(last, 48, 0, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_2'):
                skips.append(last)
                last = self.EBlock(last, 64, 1, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_3'):
                skips.append(last)
                last = self.EBlock(last, 96, 1, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_4'):
                skips.append(last)
                last = self.EBlock(last, 128, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_5'):
                skips.append(last)
                last = self.EBlock(last, 160, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_6'):
                skips.append(last)
                last = self.EBlock(last, 192, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_7'):
                skips.append(last)
                last = self.EBlock(last, 224, 3, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_8'):
                skips.append(last)
                last = self.EBlock(last, 256, 3, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            # decoder
            with tf.variable_scope('DBlock_8'):
                last = self.DBlock(last, 224, 3, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_7'):
                last = self.DBlock(last, 192, 2, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_6'):
                last = self.DBlock(last, 160, 2, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_5'):
                last = self.DBlock(last, 128, 2, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_4'):
                last = self.DBlock(last, 96, 1, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_3'):
                last = self.DBlock(last, 64, 1, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_2'):
                last = self.DBlock(last, 48, 0, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_1'):
                last = self.DBlock(last, 32, 0, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_0'):
                last = self.DBlock(last, 16, 0, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('OutBlock'):
                last = self.EBlock(last, self.in_channels, 0, [1, 8], [1, 1],
                    format, activation, normalizer, regularizer)
        # trainable/model/save/restore variables
        self.tvars = tf.trainable_variables(self.name)
        self.mvars = tf.model_variables(self.name)
        self.mvars = [i for i in self.mvars if i not in self.tvars]
        self.svars = list(set(self.tvars + self.mvars))
        self.rvars = self.svars.copy()
        # restore moving average of trainable variables
        if self.var_ema > 0:
            with tf.variable_scope('EMA'):
                self.rvars = {**{self.ema.average_name(var): var for var in self.tvars},
                    **{var.op.name: var for var in self.mvars}}
        return last

    def apply_ema(self, update_ops=[]):
        if not self.var_ema:
            return update_ops
        with tf.variable_scope('EMA'):
            with tf.control_dependencies(update_ops):
                update_ops = [self.ema.apply(self.tvars)]
            self.svars = [self.ema.average(var) for var in self.tvars] + self.mvars
        return update_ops
