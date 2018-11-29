"""
Collection of tools to help with loading and dealing with data

@author: Development Seed

utils_data.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical as to_cat

from config import model_params as MP


def get_concatenated_data(data_set_fnames, mask_type='binary', test_prop=0.2,
                          shuffle=True, max_samps=None, seed=None):
    """Helper to load a set of npz files and concatenate them."""

    if mask_type not in ['binary', 'sdt']:
        raise ValueError("`mask_type` must be 'binary' or 'sdt', got "
                         "{}".format(mask_type))

    x_datasets, y_datasets = [], []

    # Load each dataset and append
    for data_set_fname in data_set_fnames:
        data_set = np.load(data_set_fname)

        if max_samps is None:
            max_samps = data_set['x'].shape[0]

        x_datasets.append(data_set['x'][:max_samps, ...].astype(np.float32))
        if mask_type == 'binary':
            y_datasets.append(to_cat(data_set['y_binary'][:max_samps, ...],
                                     2).astype(np.bool))
        else:
            # Clip SDT to match model's output range
            c1, c2 = MP['final_layer']
            y_datasets.append(np.clip(data_set['y_sdt'][:max_samps, ...],
                                      c1, c2).astype(np.float16))
            print('\tSDT min: {}, max: {}'.format(np.min(y_datasets[-1]),
                                                  np.max(y_datasets[-1])))

        print('\tLoaded and processed {}'.format(data_set_fname))

    # Concatenate datasources into single arrays
    x_arr = np.concatenate(x_datasets)
    y_arr = np.concatenate(y_datasets)

    # Generate training/testing data
    x_train, x_test, y_train, y_test = \
        train_test_split(x_arr, y_arr, test_size=test_prop,
                         random_state=seed, shuffle=shuffle)

    return dict(x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test)
