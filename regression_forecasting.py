import argparse
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GRU, Lambda, Permute

import ists_utils
import multivariate_example
from interpolation_layer import SingleChannelInterp, CrossChannelInterp
from mimic_preprocessing import load_data, trim_los, fix_input_format
from multivariate_example import mean_imputation, hold_out


def InterpNet4Regression(*, num_features, timestamp, ref_points, hours_look_ahead, hid):
    dev = "/gpu:0"
    with tf.device(dev):
        main_input = Input(shape=(4 * num_features, timestamp), name='input')
        sci = SingleChannelInterp(ref_points, hours_look_ahead)
        cci = CrossChannelInterp()
        interp = cci(sci(main_input))
        reconst = cci(sci(main_input, reconstruction=True), reconstruction=True)
        aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
        z = Permute((2, 1))(interp)
        z = GRU(hid, activation='tanh')(z)  # no dropout, according to tensorflow documentation
        main_output = Dense(1, activation='linear', name='main_output')(z)  # regression
        orig_model = Model([main_input], [main_output, aux_output])
    model = orig_model
    print(model.summary())
    return model


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()  # Start timing at the beginning of the epoch

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()  # End timing at the end of the epoch
        elapsed_time = end_time - self.start_time
        self.epoch_times.append(elapsed_time)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training")
    ap.add_argument("-batch", "--batch_size", type=int, default=64, help="# batch size to use for training")
    ap.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs for training")
    ap.add_argument("-ref", "--reference_points", type=int, default=192, help="# of reference points")
    ap.add_argument("-units", "--hidden_units", type=int, default=100, help="# of hidden units")
    ap.add_argument("-hfadm", "--hours_from_adm", type=int, default=48, help="Hours of record to look at")  # MIMIC-III

    # ISTS arguments
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--subset', default='all')
    ap.add_argument('--dev', action='store_true', help='Run on development data')
    # ap.add_argument('--cpu', action='store_true', help='Run on CPU')
    ap.add_argument('--num-past', type=int, help='Number of past values to consider')
    ap.add_argument('--num-fut', type=int, help='Number of future values to predict')
    ap.add_argument('--nan-percentage', type=float, help='Percentage of NaN values to insert')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--abl-code', type=str, default='ES')

    args = ap.parse_args()

    gpu_num = args.gpus
    batch = args.batch_size
    epochs = args.epochs
    ref_points = args.reference_points
    hid = args.hidden_units
    hours_look_ahead = args.hours_from_adm

    dataset = args.dataset
    subset = args.subset
    num_past = args.num_past
    num_fut = args.num_fut
    nan_pct = args.nan_percentage
    seed = args.seed
    abl_code = args.abl_code

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


    if dataset == 'mimic':
        vitals, label = load_data('Dataset/mimic')
        vitals, timestamps = trim_los(vitals, hours_look_ahead)
        x, m, T = fix_input_format(vitals, timestamps)
        mean_imputation(x, m)
        x = np.concatenate((x, m, T, hold_out(m)), axis=1)  # input format
        y = np.array(label)
        print(x.shape, y.shape)
        timestamp = x.shape[2]
        num_features = x.shape[1] // 4
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        validation_args = {"validation_split": 0.2}
        loss = _multivariate_example.customloss(num_features)

    elif dataset in ["french", "ushcn", "adbpo"]:
        if args.dev:
            subset = f'{subset}_dev'
            epochs = 3
        Xy_dict, D = ists_utils.load_adapt_data("../ists/output/pickle", dataset, subset, nan_pct, num_past, num_fut, abl_code)
        X_train, X_valid, X_test = Xy_dict['X_train'], Xy_dict['X_valid'], Xy_dict['X_test']
        y_train, y_valid, y_test = Xy_dict['y_train'], Xy_dict['y_valid'], Xy_dict['y_test']
        timestamp = X_train.shape[2]
        num_features = X_train.shape[1] // 4
        validation_args = {"validation_data": ({'input': X_valid}, {'main_output': y_valid, 'aux_output': X_valid})}
        conf_name = f"{dataset}_{subset}_nan{int(nan_pct * 10)}_np{num_past}_nf{num_fut}"
        hours_look_ahead = num_past
        loss = ists_utils.customloss(num_features)


    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0., patience=20, verbose=1)
    timing_callback = TimingCallback()
    callbacks_list = [earlystop, timing_callback]


    model = InterpNet4Regression(
        num_features=num_features, 
        timestamp=timestamp, 
        ref_points=ref_points,
        hours_look_ahead=hours_look_ahead, 
        hid=hid
    )

    model.compile(
        optimizer='adam',
        loss={'main_output': 'mse', 'aux_output': loss},
        loss_weights={'main_output': 1., 'aux_output': 1.},
        metrics={'main_output': ['mse', 'mae']}
    )

    history = model.fit(
        {'input': X_train}, 
        {'main_output': y_train, 'aux_output': X_train},
        batch_size=batch,
        epochs=epochs,
        callbacks=callbacks_list,
        **validation_args,
        verbose=2,
    )

    y_pred = model.predict(X_test, batch_size=batch)
    y_pred = y_pred[0]

    if dataset == "mimic":
        metrics = model.evaluate(
            {'input': X_test},
            {'main_output': y_test, 'aux_output': X_test},
            batch_size=batch,
            verbose=0, return_dict=True
        )
        print(metrics)

    elif dataset in ["french", "ushcn", "adbpo"]:
        scalers = D['scalers']
        for id in scalers:
            for f in scalers[id]:
                if isinstance(scalers[id][f], dict):
                    scaler = StandardScaler()
                    for k, v in scalers[id][f].items():
                        setattr(scaler, k, v)
                    scalers[id][f] = scaler

        mse_test, mae_test = ists_utils.evaluate_raw(X_test, y_test, D['id_array_test'], model, D['scalers'], batch)
        mse_train, mae_train = ists_utils.evaluate_raw(X_train, y_train, D['id_array_train'], model, D['scalers'], batch)
        print('mse:', mse_test, 'mae:', mae_test)
        results_path = f'results/{conf_name}.csv'
        results = dict()
        if os.path.exists(results_path):
            results = pd.read_csv(results_path, index_col=0).to_dict(orient='index')
        results[f'{abl_code}'] = {
            'test_mae': mae_test, 'test_mse': mse_test,
            'train_mae': mae_train, 'train_mse': mse_train,
            'val_loss': history.history['val_loss'], "epoch_times": timing_callback.epoch_times
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.index.name = conf_name
        results.to_csv(results_path)
