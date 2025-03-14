import pickle

import numpy as np
from sklearn import metrics
import tensorflow as tf

from multivariate_example import hold_out


def get_X_M_T(X, feat_mask):
    def f(X):
        feat_mask_ = np.array(feat_mask)
        x_arg = feat_mask_ == 0
        m_arg = feat_mask_ == 1
        # T_arg = feat_mask_ == 2
        # X, M, T = X[:, :, x_arg], X[:, :, m_arg], X[:, :, T_arg]
        X, M = X[:, :, x_arg], X[:, :, m_arg]
        X = np.transpose(X, (0, 2, 1))
        M = np.transpose(M, (0, 2, 1))
        # T = np.transpose(T, (0, 2, 1))
        T = np.zeros_like(X)
        return X, M, T

    if len(np.shape(X)) == 3:
        return f(X)
    elif len(np.shape(X)) == 4:
        X_list = X
        X, M, T = [], [], []
        for X_ in X_list:
            x, m, t = f(X_)
            X.append(x)
            M.append(m)
            T.append(t)
        X = np.concatenate(X, axis=1)
        M = np.concatenate(M, axis=1)
        T = np.concatenate(T, axis=1)
        return X, M, T


def adapter(X, X_spt, X_exg, feat_mask, E: bool, S: bool):
    X, M, T = get_X_M_T(X, feat_mask)
    X_spt, M_spt, T_spt = get_X_M_T(X_spt, feat_mask)
    X_exg, M_exg, T_exg = get_X_M_T(X_exg, feat_mask)
    # x = np.concatenate([x, x_spt, x_exg], axis=1)
    # m = np.concatenate([m, m_spt, m_exg], axis=1)
    X, M, T = [X], [M], [T]
    if S:
        X.append(X_spt)
        M.append(M_spt)
        T.append(T_spt)
    if E:
        X.append(X_exg)
        M.append(M_exg)
        T.append(T_exg)
    X, M, T = np.concatenate(X, axis=1), np.concatenate(M, axis=1), np.concatenate(T, axis=1)
    M = 1 - M  # null indicator -> mask
    no_null_window = (M.sum(axis=-1) > 0).all(axis=-1)  # False if at least one variable in the window is entirely null, True otherwise
    X, M, T = X[no_null_window], M[no_null_window], T[no_null_window]
    T = np.apply_along_axis(lambda x: (np.arange(np.shape(x)[-1]) / np.shape(x)[-1]), -1, T)
    return X, M, T, no_null_window


def load_adapt_data(base_path, dataset, subset, nan_pct, num_past, num_fut, abl_code):
    conf_name = f"{dataset}_{subset}_nan{int(nan_pct * 10)}_np{num_past}_nf{num_fut}"
    print("Loading from", f'{base_path}/{conf_name}.pickle', "...")
    with open(f'{base_path}/{conf_name}.pickle', 'rb') as f:
        train_test_dict = pickle.load(f)
    print("Done!")

    D = train_test_dict
    feat_mask = D['x_feat_mask']
    E, S = 'E' in abl_code, 'S' in abl_code
    x_train, m_train, T_train, N_train = adapter(D['x_train'], D['spt_train'], D['exg_train'], feat_mask, E, S)
    x_valid, m_valid, T_valid, N_valid = adapter(D['x_valid'], D['spt_valid'], D['exg_valid'], feat_mask, E, S)
    x_test, m_test, T_test, N_test = adapter(D['x_test'], D['spt_test'], D['exg_test'], feat_mask, E, S)
    y_train, y_valid, y_test = D['y_train'], D['y_valid'], D['y_test']
    y_train, y_valid, y_test = y_train[N_train], y_valid[N_valid], y_test[N_test]
    # input_dim = x_train.shape[-1]

    # masked out values are set to 0
    x_train[m_train == 0] = 0
    x_valid[m_valid == 0] = 0
    x_test[m_test == 0] = 0

    x_train = np.concatenate((x_train, m_train, T_train, hold_out(m_train)), axis=1)
    x_valid = np.concatenate((x_valid, m_valid, T_valid, hold_out(m_valid)), axis=1)
    x_test = np.concatenate((x_test, m_test, T_test, hold_out(m_test)), axis=1)
    # x_train, x_valid, x_test = torch.tensor(x_train).float(), torch.tensor(x_valid).float(), torch.tensor(
    #     x_test).float()
    # y_train, y_valid, y_test = torch.tensor(y_train).float(), torch.tensor(y_valid).float(), torch.tensor(
    #     y_test).float()
    #
    X_y_dict = {
        "X_train": x_train, "X_valid": x_valid, "X_test": x_test,
        "y_train": y_train, "y_valid": y_valid, "y_test": y_test,
        # "input_dim": input_dim
    }
    params = {
        # "input_dim": input_dim,
        "scalers": D['scalers'],
        "id_array_train": D['id_train'],
        "id_array_valid": D['id_valid'],
        "id_array_test": D['id_test']
    }
    return X_y_dict, params


def evaluate_raw(X, y, id_array, model, scalers, batch_size):
    y_true = y
    y_pred, _ = model.predict(X, batch_size=batch_size)
    y_pred, y_true = np.reshape(y_pred, (-1, 1)), np.reshape(y_true, (-1, 1))
    y_true = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_true, id_array)])
    y_pred = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_pred, id_array)])
    mse, mae = metrics.mean_squared_error(y_true, y_pred), metrics.mean_absolute_error(y_true, y_pred)
    return mse, mae


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
