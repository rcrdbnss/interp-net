import pickle

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.src.layers import Lambda, Permute, GRU
from tensorflow.keras.layers import Dense

from interpolation_layer import SingleChannelInterp, CrossChannelInterp


def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we removecthe data point as an input to the interpolation network,
    and includecthe predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(perc * count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(perc * count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask


def mean_imputation(vitals, mask):
    """For the time series missing entirely, our interpolation network
    assigns the starting point (time t=0) value of the time series to
    the global mean before applying the two-layer interpolation network.
    In such cases, the first interpolation layer just outputs the global
    mean for that channel, but the second interpolation layer performs
    a more meaningful interpolation using the learned correlations from
    other channels."""
    counts = np.sum(np.sum(mask, axis=2), axis=0)
    mean_values = np.sum(np.sum(vitals * mask, axis=2), axis=0) / counts
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.sum(mask[i, j]) == 0:
                mask[i, j, 0] = 1
                vitals[i, j, 0] = mean_values[j]
    return


def customloss(num_features):
    def f(ytrue, ypred):
        """ Autoencoder loss
        """
        y = ytrue[:, :num_features, :]
        m2 = ytrue[:, 3*num_features:4*num_features, :]
        m2 = 1 - m2
        m1 = ytrue[:, num_features:2*num_features, :]
        m = m1 * m2
        ypred = ypred[:, :num_features, :]
        x = (y - ypred) * (y - ypred)
        x = x * m
        count = tf.reduce_sum(m, axis=2)
        count = tf.where(count > 0, count, tf.ones_like(count))
        x = tf.reduce_sum(x, axis=2) / count
        x = tf.reduce_sum(x, axis=1) / num_features
        loss = tf.reduce_mean(x)
        return loss
    return f


def _customloss(num_features):
    def f(ytrue, ypred):
        """ Autoencoder loss
        """
        # standard deviation of each feature mentioned in paper for MIMIC_III data
        wc = np.array([3.33, 23.27, 5.69, 22.45, 14.75, 2.32,
                       3.75, 1.0, 98.1, 23.41, 59.32, 1.41])
        wc.shape = (1, num_features)
        y = ytrue[:, :num_features, :]
        m2 = ytrue[:, 3 * num_features:4 * num_features, :]
        m2 = 1 - m2
        m1 = ytrue[:, num_features:2 * num_features, :]
        m = m1 * m2
        ypred = ypred[:, :num_features, :]
        x = (y - ypred) * (y - ypred)
        x = x * m
        count = tf.reduce_sum(m, axis=2)
        count = tf.where(count > 0, count, tf.ones_like(count))
        x = tf.reduce_sum(x, axis=2) / count
        x = x / (wc ** 2)  # dividing by standard deviation
        x = tf.reduce_sum(x, axis=1) / num_features
        return tf.reduce_mean(x)
    return f


def interp_net(*, num_features, timestamp, ref_points, hours_look_ahead, hid):
    # if gpu_num > 1:
    #     dev = "/cpu:0"
    # else:
    #     dev = "/gpu:0"
    dev = "/gpu:0"
    with tf.device(dev):
        main_input = Input(shape=(4 * num_features, timestamp), name='input')
        sci = SingleChannelInterp(ref_points, hours_look_ahead)
        cci = CrossChannelInterp()
        interp = cci(sci(main_input))
        reconst = cci(sci(main_input, reconstruction=True), reconstruction=True)

        aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
        z = Permute((2, 1))(interp)
        # z = GRU(hid, activation='tanh', recurrent_dropout=0.2, dropout=0.2)(z)
        z = GRU(hid, activation='tanh')(z)
        # main_output = Dense(1, activation='sigmoid', name='main_output')(z)  # classification
        main_output = Dense(1, activation='linear', name='main_output')(z)  # regression
        orig_model = Model([main_input], [main_output, aux_output])
    # if gpu_num > 1:
    #     model = multi_gpu_model(orig_model, gpus=gpu_num)
    # else:
    #     model = orig_model
    model = orig_model
    print(orig_model.summary())
    return model


def model_step(X_train, X_valid, y_train, y_valid, *,
               ref_points, hours_look_ahead, hid,
               batch_size, epochs, callbacks, eager=False):
    timestamps = X_train.shape[2]
    num_features = X_train.shape[1] // 4
    model = interp_net(
        num_features=num_features,
        timestamp=timestamps,
        ref_points=ref_points,
        hours_look_ahead=hours_look_ahead,
        hid=hid
    )
    model.compile(
        optimizer='adam',
        loss={'main_output': 'mse', 'aux_output': customloss(num_features)},
        loss_weights={'main_output': 1., 'aux_output': 1.},
        metrics={'main_output': ['mse', 'mae']},
        run_eagerly=eager,
        # run_eagerly=False,
    )
    history = model.fit(
        {'input': X_train}, {'main_output': y_train, 'aux_output': X_train},
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=({'input': X_valid}, {'main_output': y_valid, 'aux_output': X_valid}),
        verbose=1,
        shuffle=False
    )
    return history.model, history.history
