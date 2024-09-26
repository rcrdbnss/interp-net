import numpy as np
import tensorflow as tf

from mimic_preprocessing import load_data, trim_los, fix_input_format
from mvar import mean_imputation


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

hours_look_ahead = 48

vitals, label = load_data('Dataset/mimic/')
vitals, timestamps = trim_los(vitals, hours_look_ahead)
x, m, T = fix_input_format(vitals, timestamps)
mean_imputation(x, m)
num_features = x.shape[1]
x = np.concatenate((x, m, T), axis=1)  # input format
y = np.array(label)

with open('Dataset/mimic/X.npy', 'wb') as f:
    np.save(f, x)
with open('Dataset/mimic/y.npy', 'wb') as f:
    np.save(f, y)

