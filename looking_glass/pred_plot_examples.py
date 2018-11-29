"""
Run inference with model and plot some example results.

@author: developmentseed

pred_plot_examples.py
"""
import os.path as op

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils import multi_gpu_model

from model_dlab import preprocess_input, BilinearUpsampling
from config import (inference_plot_params as IP, pred_params as PP,
                    model_params as MP, ckpt_dir, preds_dir)
from utils_training import load_model, check_create_dir
from utils_dataflow import get_concatenated_data


def plot_prediction(x_samp, mask_gt, mask_p, save_path, mask_type):
    """"""
    plt.close('all')

    titles = ['Raster image', 'Ground truth', 'Predicted Mask']
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].imshow(x_samp.astype(np.uint8))
    axes[0].set_title(titles[0], fontsize=18)
    axes[0].axis('off')

    for ax, img, title in zip(axes[1:], [mask_gt, mask_p], titles[1:]):
        if mask_type == 'sdt':
            cl, ch = MP['final_layer']
            img_normed = (img - cl) / (ch - cl)
            ax.imshow(plt.cm.bwr(img_normed), cmap='bwr', vmin=0., vmax=1.)
        else:
            im = ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=18)
        ax.axis('off')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()


def plot_example_batch(model, save_dir, mask_type, n_examples=5, seed=None):
    """Calculate inference on a set of tiles"""

    sqm_per_pix = IP['img_res'] ** 2
    np.random.seed(seed)
    img_batch_inds = np.random.choice(len(data_dict['x_test']), IP['n_plots'],
                                      replace=False)
    x_batch = data_dict['x_test'][img_batch_inds, ...]
    y_batch = data_dict['y_test'][img_batch_inds, ...]

    # Run prediction with model and get m^2
    y_p = model.predict(preprocess_input(x_batch.copy().astype(np.float)),
                        batch_size=n_examples, verbose=0)

    y_p_masks = y_p[..., -1]
    total_sq_m = sqm_per_pix * np.sum(y_p_masks > 0,
                                      axis=tuple(range(1, y_p_masks.ndim)))

    for it, (x_img, y_p_mask, y_mask) in enumerate(zip(x_batch, y_p_masks, y_batch)):
        save_path = op.join(save_dir, 'example_{:03d}.png'.format(it))
        x_img = x_img.reshape((256, 256, 3))
        y_mask = y_mask.reshape((256, 256)).squeeze()
        y_p_mask = y_p_mask.reshape((256, 256))
        plot_prediction(x_img, y_mask, y_p_mask, save_path, mask_type)

        if it % 10 == 0:
            print('Predicted and plotted {} samples.'.format(it))


if __name__ == "__main__":
    ####################################
    # Load image set
    ####################################
    # Loading data can take a while, check if it exists in interactive ipython session
    try:
        data_dict
        print('Found previous `data_dict` variable.')
    except NameError:
        print('Unpacking data for plotting.')
        # Run data straight from label-maker
        if IP['data_type'] == 'label-maker':
            data_npz = np.load(IP['data_set_fpath'])
            data_dict = {}
            for key in data_npz.keys():
                data_dict[key] = data_npz[key]

        # Run data from spacenet
        elif IP['data_type'] == 'spacenet':
            data_dict = get_concatenated_data([IP['data_set_fpath']], IP['mask_type'],
                                              test_prop=MP['test_prop'],
                                              shuffle=False,
                                              max_samps=MP['max_sing_dataset_samps'],
                                              seed=MP['shuffle_seed'])
            c_dim = 2 if MP['mask_type'] == 'binary' else 1
            data_dict['y_train'] = data_dict['y_train'].reshape(
                (-1, MP['input_shape'][0] * MP['input_shape'][1], c_dim))
            data_dict['y_test'] = data_dict['y_test'].reshape(
                (-1, MP['input_shape'][0] * MP['input_shape'][1], c_dim))
        else:
            raise ValueError('Incorrect specification for "data_type"')

    print('Found {} samples for training and {} for testing.'.format(
        data_dict['x_train'].shape[0], data_dict['x_test'].shape[0]))

    ####################################
    # Load model and params
    ####################################
    if PP['n_gpus'] > 1:
        # Load weights on CPU to avoid taking up GPU space
        with tf.device('/cpu:0'):
            template_model = load_model(op.join(ckpt_dir, PP['model_arch_fname']),
                                        op.join(ckpt_dir, PP['model_weights_fname']),
                                        custom_objects={'BilinearUpsampling':BilinearUpsampling})
        model = multi_gpu_model(template_model, gpus=PP['n_gpus'])
    else:
        template_model = load_model(op.join(ckpt_dir, PP['model_arch_fname']),
                                    op.join(ckpt_dir, PP['model_weights_fname']),
                                    custom_objects={'BilinearUpsampling':BilinearUpsampling})
        model = template_model

    # Turn off training. This is supposed to be faster (haven't seen this empirically though)
    K.set_learning_phase = 0
    for layer in template_model.layers:
        layer.trainable = False

    check_create_dir(IP['save_dir'])
    plot_example_batch(model, save_dir=IP['save_dir'],
                       mask_type=IP['mask_type'], n_examples=IP['n_plots'])
    print('Processed {} images'.format(IP['n_plots']))
