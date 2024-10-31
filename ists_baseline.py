import argparse
import os.path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

import ists_utils
from mvar import model_step


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

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subset', default='all')
    parser.add_argument('--dev', action='store_true', help='Run on development data')
    # parser.add_argument('--cpu', action='store_true', help='Run on CPU')
    parser.add_argument('--num-past', type=int, required=True, help='Number of past values to consider')
    parser.add_argument('--num-fut', type=int, required=True, help='Number of future values to predict')
    parser.add_argument('--nan-percentage', type=float, required=True, help='Percentage of NaN values to insert')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--abl-code', type=str, default='ES')
    parser.add_argument("-batch", "--batch-size", type=int, default=256, help="# batch size to use for training")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs for training")
    parser.add_argument("-units", "--hidden-units", type=int, default=100, help="# of hidden units")
    args = parser.parse_args()
    
    dataset = args.dataset
    subset = args.subset
    num_past = args.num_past
    num_fut = args.num_fut
    nan_pct = args.nan_percentage
    seed = args.seed
    abl_code = args.abl_code
    batch = args.batch_size
    epochs = args.epochs
    hid = args.hidden_units
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if args.dev:
        subset = f'{subset}_dev'
        epochs = 3

    conf_name = f"{dataset}_{subset}_nan{int(nan_pct * 10)}_np{num_past}_nf{num_fut}"

    Xy_dict, D = ists_utils.load_adapt_data(None, dataset, subset, nan_pct, num_past, num_fut, abl_code)
    x_train, x_valid, x_test = Xy_dict['X_train'], Xy_dict['X_valid'], Xy_dict['X_test']
    y_train, y_valid, y_test = Xy_dict['y_train'], Xy_dict['y_valid'], Xy_dict['y_test']

    hours_look_ahead = num_past
    ref_points = num_past

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0000, patience=20, mode='min', verbose=1, restore_best_weights=True
    )
    callbacks_list = [earlystop]

    model, history = model_step(x_train, x_valid, y_train, y_valid,
                    ref_points=ref_points, hours_look_ahead=hours_look_ahead, hid=hid,
                    batch_size=batch, epochs=epochs, callbacks=callbacks_list, eager=args.dev)

    scalers = D['scalers']

    mse_test, mae_test = ists_utils.evaluate_raw(x_test, y_test, D['id_array_test'], model, scalers, batch)
    mse_train, mae_train = ists_utils.evaluate_raw(x_train, y_train, D['id_array_train'], model, scalers, batch)

    print('mse:', mse_test, 'mae:', mae_test)

    results_path = f'results/{conf_name}.csv'
    results = dict()
    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col=0).to_dict(orient='index')
    results[f'{abl_code}'] = {
        'test_mae': mae_test, 'test_mse': mse_test,
        'train_mae': mae_train, 'train_mse': mse_train,
        'val_loss': history['val_loss']
    }
    results = pd.DataFrame.from_dict(results, orient='index')
    results.index.name = conf_name
    results.to_csv(results_path)
