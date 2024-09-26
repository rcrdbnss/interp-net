import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
import numpy as np


class SingleChannelInterp(Layer):

    def __init__(self, ref_points, hours_look_ahead, **kwargs):
        self.ref_points = ref_points
        self.hours_look_ahead = hours_look_ahead  # in hours
        super(SingleChannelInterp, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape [batch, features, time_stamp]
        self.time_stamp = input_shape[2]
        self.d_dim = input_shape[1] // 4
        self.activation = activations.get('sigmoid')
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.d_dim,),
            initializer=tf.keras.initializers.Constant(value=0.0),
            trainable=True)
        super(SingleChannelInterp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        x_t = x[:, :self.d_dim, :]
        d = x[:, 2 * self.d_dim:3 * self.d_dim, :]
        if reconstruction:
            output_dim = self.time_stamp
            m = x[:, 3 * self.d_dim:, :]
            ref_t = tf.tile(d[:, :, None, :], (1, 1, output_dim, 1))
        else:
            m = x[:, self.d_dim: 2 * self.d_dim, :]
            ref_t = np.linspace(0, self.hours_look_ahead, self.ref_points)
            output_dim = self.ref_points
            ref_t = tf.convert_to_tensor(ref_t, dtype=tf.float32)
            ref_t = tf.reshape(ref_t, (1, ref_t.shape[0]))
        d = tf.tile(d[:, :, :, None], (1, 1, 1, output_dim))
        mask = tf.tile(m[:, :, :, None], (1, 1, 1, output_dim))
        x_t = tf.tile(x_t[:, :, :, None], (1, 1, 1, output_dim))
        norm = (d - ref_t) * (d - ref_t)
        a = tf.ones((self.d_dim, self.time_stamp, output_dim))
        pos_kernel = tf.math.log(1 + tf.exp(self.kernel))
        alpha = a * pos_kernel[:, tf.newaxis, tf.newaxis]
        w = tf.math.reduce_logsumexp(-alpha * norm + tf.math.log(mask), axis=2)
        w1 = tf.tile(w[:, :, None, :], (1, 1, self.time_stamp, 1))
        w1 = tf.exp(-alpha * norm + tf.math.log(mask) - w1)
        y = tf.reduce_sum(w1 * x_t, axis=2)
        if reconstruction:
            rep1 = tf.concat([y, w], 1)
        else:
            w_t = tf.math.reduce_logsumexp(-10.0 * alpha * norm + tf.math.log(mask), axis=2)  # kappa = 10

            w_t = tf.tile(w_t[:, :, None, :], (1, 1, self.time_stamp, 1))
            w_t = tf.exp(-10.0 * alpha * norm + tf.math.log(mask) - w_t)
            y_trans = tf.reduce_sum(w_t * x_t, axis=2)
            rep1 = tf.concat([y, w, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], 2 * self.d_dim, self.time_stamp)
        return (input_shape[0], 3 * self.d_dim, self.ref_points)


class CrossChannelInterp(Layer):

    def __init__(self, **kwargs):
        super(CrossChannelInterp, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d_dim = input_shape[1] // 3
        self.activation = activations.get('sigmoid')
        self.cross_channel_interp = self.add_weight(
            name='cross_channel_interp',
            shape=(self.d_dim, self.d_dim),
            initializer=tf.keras.initializers.Identity(gain=1.0),
            trainable=True)

        super(CrossChannelInterp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        self.output_dim = K.int_shape(x)[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2 * self.d_dim, :]
        intensity = tf.exp(w)
        y = tf.transpose(y, perm=[0, 2, 1])
        w = tf.transpose(w, perm=[0, 2, 1])
        w2 = w
        w = tf.tile(w[:, :, :, None], (1, 1, 1, self.d_dim))
        den = tf.math.reduce_logsumexp(w, axis=2)
        w = tf.exp(w2 - den)
        mean = tf.reduce_mean(y, axis=1)
        mean = tf.tile(mean[:, None, :], (1, self.output_dim, 1))
        w2 = tf.linalg.matmul(w * (y - mean), cross_channel_interp) + mean
        rep1 = tf.transpose(w2, perm=[0, 2, 1])
        if reconstruction is False:
            y_trans = x[:, 2 * self.d_dim:3 * self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = tf.concat([rep1, intensity, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], self.d_dim, self.output_dim)
        return (input_shape[0], 3 * self.d_dim, self.output_dim)
