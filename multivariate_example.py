import argparse
import logging
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from mvar import hold_out, mean_imputation, model_step

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

from mimic_preprocessing import load_data, trim_los, fix_input_format
import warnings

warnings.filterwarnings("ignore")

np.random.seed(10)
tf.random.set_seed(10)


def data_step(*, hours_look_ahead):
    vitals, label = load_data('Dataset/mimic/')
    vitals, timestamps = trim_los(vitals, hours_look_ahead)
    x, m, T = fix_input_format(vitals, timestamps)
    mean_imputation(x, m)
    num_features = x.shape[1]
    x = np.concatenate((x, m, T, hold_out(m)), axis=1)  # input format
    y = np.array(label)

    test_size = 0.2
    test_start = int(len(y) * (1 - test_size))
    X_train, X_test, y_train, y_test = x[:test_start], x[test_start:], y[:test_start], y[test_start:]
    valid_size = 0.2
    valid_start = int(len(y_train) * (1 - valid_size))
    X_train, X_valid, y_train, y_valid = X_train[:valid_start], X_train[valid_start:], y_train[:valid_start], y_train[valid_start:]
    for i in range(num_features):
        scaler = StandardScaler()
        X_train[:, i, :] = scaler.fit_transform(X_train[:, i, :])
        X_valid[:, i, :] = scaler.transform(X_valid[:, i, :])
        X_test[:, i, :] = scaler.transform(X_test[:, i, :])
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_valid = y_scaler.transform(y_valid.reshape(-1, 1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    return X_train, X_valid, X_test, y_train, y_valid, y_test, y_scaler


if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training")

    ap.add_argument("-batch", "--batch_size", type=int, default=256, help="# batch size to use for training")

    ap.add_argument("-e", "--epochs", type=int, default=2, help="# of epochs for training")

    ap.add_argument("-ref", "--reference_points", type=int, default=192, help="# of reference points")

    ap.add_argument("-units", "--hidden_units", type=int, default=100, help="# of hidden units")

    ap.add_argument("-hfadm", "--hours_from_adm", type=int, default=48, help="Hours of record to look at")

    args = vars(ap.parse_args())
    gpu_num = args["gpus"]
    epoch = args["epochs"]
    hid = args["hidden_units"]
    ref_points = args["reference_points"]
    hours_look_ahead = args["hours_from_adm"]
    batch = args["batch_size"]

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0000, patience=20, verbose=0)
    callbacks_list = [earlystop]

    # Loading dataset
    X_train, X_valid, X_test, y_train, y_valid, y_test, y_scaler = data_step(hours_look_ahead=hours_look_ahead)

    model = model_step(X_train, X_valid, y_train, y_valid,
                       ref_points=ref_points, hours_look_ahead=hours_look_ahead, hid=hid,
                       batch_size=batch, epochs=epoch, callbacks=callbacks_list)

    y_pred = model.predict(X_test, batch_size=batch)
    y_pred = y_scaler.inverse_transform(y_pred[0]).reshape(-1)
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    mse, mae = mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)
    print('mse:', mse, 'mae:', mae)
