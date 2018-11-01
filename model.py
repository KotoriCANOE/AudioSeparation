import tensorflow as tf
import layers
from network import Generator

DATA_FORMAT = 'NCHW'

class Model:
    def __init__(self, config):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.in_channels = 2
        self.out_channels = 2
        # collections
        self.train_sums = []
        self.loss_sums = []
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
        # internal parameters
        self.input_shape = [None, None, None, None]
        self.input_shape[-3 if self.data_format == 'NCHW' else -1] = self.in_channels
        self.output_shape = self.input_shape

    def build_model(self, inputs=None):
        # inputs
        if inputs is None:
            self.inputs = tf.placeholder(self.dtype, self.input_shape, name='Input')
        else:
            self.inputs = tf.identity(inputs, name='Input')
            self.inputs.set_shape(self.input_shape)
        # forward pass
        self.generator = Generator('Generator', self.config)
        self.outputs = self.generator(self.inputs, reuse=None)
        # outputs
        self.outputs = tf.identity(self.outputs, name='Output')
        # all the saver variables
        self.svars = self.generator.svars
        # all the restore variables
        self.rvars = self.generator.rvars
        # return outputs
        return self.outputs

    def build_train(self, inputs=None, labels=None):
        # style reference
        if labels is None:
            self.labels = tf.placeholder(self.dtype, self.output_shape, name='Label')
        else:
            self.labels = tf.identity(labels, name='Label')
            self.labels.set_shape(self.output_shape)
        # build model
        self.build_model(inputs)
        # build loss
        self.build_g_loss(self.outputs, self.labels)

    def build_g_loss(self, outputs, labels):
        self.g_log_losses = []
        update_ops = []
        loss_key = 'GeneratorLoss'
        with tf.variable_scope(loss_key):
            # L1 loss
            l1_loss = tf.losses.absolute_difference(labels, outputs)
            update_ops.append(self.loss_summary('l1_loss', l1_loss, self.g_log_losses))
            # MS-SSIM loss
            ssim_loss = tf.constant(0, tf.float32)
            for label, output in zip(tf.split(labels, 2, axis=-3), tf.split(outputs, 2, axis=-3)):
                ssim_loss += 1 - layers.MS_SSIM(label + 1, output + 1, L=2,
                    radius=10, sigma=4.0, data_format=self.data_format, one_dim=True)
            tf.losses.add_loss(ssim_loss)
            update_ops.append(self.loss_summary('ssim_loss', ssim_loss, self.g_log_losses))
            # total loss
            losses = tf.losses.get_losses(loss_key)
            main_loss = tf.add_n(losses, 'main_loss')
            # regularization loss - weight
            reg_losses = tf.losses.get_regularization_losses('Generator')
            reg_loss = tf.add_n(reg_losses)
            update_ops.append(self.loss_summary('reg_loss', reg_loss))
            # final loss
            self.g_loss = main_loss + reg_loss
            update_ops.append(self.loss_summary('loss', self.g_loss))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.g_losses_acc = tf.no_op('accumulator')

    def train(self, global_step):
        model = self.generator
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Generator')
        # learning rate
        lr = tf.train.cosine_decay_restarts(1e-3,
            global_step, 1000, t_mul=2.0, m_mul=0.85, alpha=1e-1)
        lr = tf.train.exponential_decay(lr, global_step, 1000, 0.99)
        self.train_sums.append(tf.summary.scalar('Generator/LR', lr))
        # optimizer
        opt = tf.contrib.opt.NadamOptimizer(lr)
        with tf.control_dependencies(update_ops):
            grads_vars = opt.compute_gradients(self.g_loss, model.tvars)
            update_ops = [opt.apply_gradients(grads_vars, global_step)]
        # histogram for gradients and variables
        for grad, var in grads_vars:
            self.train_sums.append(tf.summary.histogram(var.op.name + '/grad', grad))
            self.train_sums.append(tf.summary.histogram(var.op.name, var))
        # save moving average of trainalbe variables
        update_ops = model.apply_ema(update_ops)
        # all the saver variables
        self.svars = self.generator.svars
        # return optimizing op
        with tf.control_dependencies(update_ops):
            train_op = tf.no_op('train')
        return train_op

    def loss_summary(self, name, loss, collection=None):
        with tf.variable_scope('LossSummary/' + name):
            # internal variables
            loss_sum = tf.get_variable('sum', (), tf.float32, tf.initializers.zeros(tf.float32))
            loss_count = tf.get_variable('count', (), tf.float32, tf.initializers.zeros(tf.float32))
            # accumulate to sum and count
            acc_sum = loss_sum.assign_add(loss, True)
            acc_count = loss_count.assign_add(1.0, True)
            # calculate mean
            loss_mean = tf.divide(loss_sum, loss_count, 'mean')
            if collection is not None:
                collection.append(loss_mean)
            # reset sum and count
            with tf.control_dependencies([loss_mean]):
                clear_sum = loss_sum.assign(0.0, True)
                clear_count = loss_count.assign(0.0, True)
            # log summary
            with tf.control_dependencies([clear_sum, clear_count]):
                self.loss_sums.append(tf.summary.scalar('value', loss_mean))
            # return after updating sum and count
            with tf.control_dependencies([acc_sum, acc_count]):
                return tf.identity(loss, 'loss')

    def get_summaries(self):
        train_summary = tf.summary.merge(self.train_sums) if self.train_sums else None
        loss_summary = tf.summary.merge(self.loss_sums) if self.loss_sums else None
        return train_summary, loss_summary
