"""
train_deeplab.py

@author: Development Seed

Train the DeepLabV3+ network to segment buildings
"""

import os
from os import path as op
from functools import partial
from datetime import datetime as dt
import pickle
import pprint

import yaml
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import AddSignOptimizer, PowerSignOptimizer
from keras import backend as K
from keras.optimizers import Adam, rmsprop, SGD, TFOptimizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (ModelCheckpoint, EarlyStopping, TensorBoard,
                             ReduceLROnPlateau)
from keras.utils import plot_model
from keras.utils import multi_gpu_model
#from tensorflow.train import linear_cosine_decay, noisy_linear_cosine_decay
from hyperopt import fmin, Trials, STATUS_OK, tpe

from config import (get_params, tboard_dir, ckpt_dir, model_params as MP,
                    img_aug_params as IMP, save_params as SP)
#from utils_metrics import get_lr_metric
from utils_training import (print_start_details, print_end_details,
                            TensorBoardImage, sync_models_to_S3,
                            sync_tb_dirs_to_S3)
from utils_dataflow import get_concatenated_data
from model_dlab import get_deepLabV3p, preprocess_input


def get_optimizer(opt_params, lr):
    """Helper to get optimizer from text params"""
    if opt_params['opt_func'] == 'sgd':
        return SGD(lr=lr, momentum=opt_params['momentum'])
    elif opt_params['opt_func'] == 'adam':
        return Adam(lr=lr)
    elif opt_params['opt_func'] == 'rmsprop':
        return rmsprop(lr=lr)
    elif opt_params['opt_func'] == 'powersign':
        from tensorflow.contrib.opt.python.training import sign_decay as sd
        d_steps = opt_params['decay_steps']
        # Define the decay function (if specified)
        if opt_params['decay_func'] == 'lin':
            decay_func = sd.get_linear_decay_fn(d_steps)
        elif opt_params['decay_func'] == 'cos':
            decay_func = sd.get_consine_decay_fn(d_steps)
        elif opt_params['decay_func'] == 'res':
            decay_func = sd.get_restart_decay_fn(d_steps,
                                                 num_periods=opt_params['decay_periods'])
        elif opt_params['decay_func'] is None:
            decay_func = None
        else:
            raise ValueError('decay function not specified correctly')

        # Use decay function in TF optimizer
        return TFOptimizer(PowerSignOptimizer(learning_rate=lr,
                                              sign_decay_fn=decay_func))
    else:
        raise ValueError('Optimizer specification not understood')


def run_deeplab_trial(params):
    """Train DeepLabV3+ network and return score for hyperopt

    Parmeters:
    ----------
    params: dict
        Parameters returned from config.get_params() for hyperopt

    Returns:
    --------
    result_dict: dict
        Results of model training for hyperopt.
    """

    K.clear_session()  # Remove any existing graphs
    mst_str = dt.now().strftime("%m%d_%H%M%S")

    print('\n' + '=' * 40 + '\nStarting model at {}'.format(mst_str))
    print('Model # %s' % len(trials))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    ######################
    # Paths and Callbacks
    ######################
    ckpt_fpath = op.join(ckpt_dir, mst_str + \
        '_MSE{val_loss:02.2f}_E{epoch:02d}_weights.h5')
    tboard_model_dir = op.join(tboard_dir, mst_str)

    callbacks_0 = [TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                               write_grads=False),
                   TensorBoardImage('Prediction', tboard_model_dir,
                                    pred_tboard_examples, MP['mask_type'])]
    callbacks_1plus = [
        TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                    write_grads=False),
        ModelCheckpoint(ckpt_fpath, monitor='val_loss',
                        save_weights_only=True, save_best_only=True),
        #EarlyStopping(min_delta=MP['early_stopping_min_delta'],
        #              patience=MP['early_stopping_patience'], verbose=1)]
        #ReduceLROnPlateau(epsilon=MP['reduce_lr_epsilon'],
        #                  patience=MP['reduce_lr_patience'], verbose=1),
        TensorBoardImage('Prediction', tboard_model_dir, pred_tboard_examples,
                         MP['mask_type'])]

    #########################
    # Construct model
    #########################

    if MP['n_gpus'] == 1:
        template_model = get_deepLabV3p(MP['input_shape'], MP['num_classes'],
                                        MP['final_layer'], MP['output_stride'])
        # Load the pre-trained DeepLab_v3+ weights if specified
        if MP['load_orig_weights']:
            template_model.load_weights(MP['orig_weights_fpath'], by_name=True)
    else:
        # Load weights on CPU to avoid taking up GPU space
        with tf.device('/cpu:0'):
            template_model = get_deepLabV3p(MP['input_shape'], MP['num_classes'],
                                            MP['final_layer'], MP['output_stride'])
            # Load the pre-trained DeepLab_v3+ weights if specified
            if MP['load_orig_weights']:
                template_model.load_weights(MP['orig_weights_fpath'], by_name=True)

    if MP['n_gpus'] == 1:
        model = template_model
    else:
        # Don't add `cpu_relocation=True` here if using `with cpu` above. Causes OOM
        model = multi_gpu_model(template_model, gpus=MP['n_gpus'])

    # Print layer names/indices:
    #for li, layer in enumerate(template_model.layers):
    #    print(li, layer.name)

    #############################
    # Save model training details
    #############################
    model_yaml = template_model.to_yaml()
    save_template = op.join(ckpt_dir, mst_str + '_{}.{}')
    arch_fpath = save_template.format('arch', 'yaml')
    if not op.exists(arch_fpath):
        with open(arch_fpath.format('arch', 'yaml'), 'w') as yaml_file:
            yaml_file.write(model_yaml)

    # Save params to yaml file
    params_fpath = save_template.format('params', 'yaml')
    if not op.exists(params_fpath):
        with open(params_fpath, 'w') as yaml_file:
            yaml_file.write(yaml.dump(params))
            yaml_file.write(yaml.dump(MP))

    #plot_model(template_model, to_file='model_schematic.png')

    ####################
    # Train top layers
    ####################
    # TODO: better way to directly access epochs finished?
    epochs_finished = 0

    for pi, (ph_steps, frz_cutoff, ph_lr) in enumerate(MP['phase_lengths']):

        print('\nPhase {}, layer cutoff: {}, learning rate: {}'.format(
            pi, frz_cutoff, ph_lr))
        # Train the top layers only by setting all lower layers untrainable
        for layer in template_model.layers:
            layer.trainable = False
        if frz_cutoff == 'batch_norm':
            for layer in template_model.layers:
                if '_BN' in layer.name:
                    layer.trainable = True
        elif isinstance(frz_cutoff, str):
            if 'batch_norm_freeze' in frz_cutoff:
                new_frz_cutoff = int(frz_cutoff.split()[1])
                for layer in template_model.layers[new_frz_cutoff:]:
                    if '_BN' in layer.name:
                        continue
                    else:
                        layer.trainable = True
        else:
            for layer in template_model.layers[frz_cutoff:]:
                layer.trainable = True

        # Compile the model (do this after setting non-trainable layers)
        #lr_metric = get_lr_metric(optimizer)
        model.compile(optimizer=get_optimizer(params['optimizer'], lr=ph_lr),
                      loss=params['loss'], metrics=MP['metrics'])

        # Define template for saving/callback eval
        model.__setattr__('callback_model', template_model)

        hist = model.fit_generator(
            train_gen.flow(data_dict['x_train'], data_dict['y_train'],
                           batch_size=MP['batch_size'] * MP['n_gpus']),
            steps_per_epoch=steps_per_epo,
            epochs=epochs_finished + ph_steps,
            # TODO: larget batch size possible on inference? Dif from training?
            validation_data=test_gen.flow(data_dict['x_test'], data_dict['y_test'],
                                          batch_size=MP['batch_size'] * MP['n_gpus']),
            validation_steps=steps_per_val,
            callbacks=callbacks_0 if pi == 0 else callbacks_1plus,
            initial_epoch=epochs_finished,
            verbose=1)
        epochs_finished += ph_steps

        if SP['save_models_to_S3']:
            sync_models_to_S3()
            sync_tb_dirs_to_S3()

    # Return best score of last validation accuracy batch
    check_ind = -1 * (MP['early_stopping_patience'] + 1)
    result_dict = dict(loss=np.min(hist.history['val_loss'][check_ind:]),
                       status=STATUS_OK)

    return result_dict


if __name__ == '__main__':
    start_time = dt.now()
    print_start_details(start_time)

    #########################
    # Load data
    #########################
    print('Loading prediction images for tensorboard.')
    # Load images for prediction that will appear as examples in tensorboard
    pred_img_fnames = [fname for fname in os.listdir(tboard_dir) if
                       op.splitext(fname)[1] == '.png']
    pred_tboard_examples = np.zeros([len(pred_img_fnames)] + list(MP['input_shape']))
    for fi, fname_pred_img in enumerate(pred_img_fnames):
        temp_img = Image.open(op.join(tboard_dir, fname_pred_img))
        pred_tboard_examples[fi, :, :, :] = np.array(temp_img, np.float)
    pred_tboard_examples = preprocess_input(pred_tboard_examples)

    # Loading data can take a while, check if it exists in interactive ipython session
    try:
        data_dict.keys()
        print('Found previous `data_dict` variable.')
        train_gen, test_gen = None, None
        trials = None
    except NameError:
        print('Unpacking training data.')
        training_datasets = [op.join(trn_dir, 'data.npz') for trn_dir in
                             MP['training_data_dirs']]
        data_dict = get_concatenated_data(training_datasets, MP['mask_type'],
                                          test_prop=MP['test_prop'], shuffle=True,
                                          max_samps=MP['max_sing_dataset_samps'],
                                          seed=MP['shuffle_seed'])
        # Apply preproc once up front instead of in image data generator
        if MP['tf_preproc']:
            data_dict['x_train'] = preprocess_input(data_dict['x_train']).astype(np.float16)
            data_dict['x_test'] = preprocess_input(data_dict['x_test']).astype(np.float16)

        c_dim = 2 if MP['mask_type'] == 'binary' else 1
        data_dict['y_train'] = data_dict['y_train'].reshape(
            (-1, MP['input_shape'][0] * MP['input_shape'][1], c_dim))
        data_dict['y_test'] = data_dict['y_test'].reshape(
            (-1, MP['input_shape'][0] * MP['input_shape'][1], c_dim))

    print('Found {} samples for training and {} for testing.'.format(
        data_dict['x_train'].shape[0], data_dict['x_test'].shape[0]))

    # Define train/test steps per epoch
    steps_per_epo = (len(data_dict['x_train']) * MP['prop_total_img_set']) // \
        (MP['batch_size'] * MP['n_gpus'])
    steps_per_val = len(data_dict['x_test']) // (MP['batch_size'] * MP['n_gpus'])

    #########################
    # Generate the generators
    #########################
    train_gen = ImageDataGenerator(**IMP)
    test_gen = ImageDataGenerator()

    ###############################
    # Define Hyperopt optimization
    ###############################
    trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=MP['n_rand_hp_iters'])
    argmin = fmin(run_deeplab_trial, space=get_params(MP), algo=algo,
                  max_evals=MP['n_total_hp_iters'], trials=trials)

    ###############################
    # End of training cleanup
    ###############################
    end_time = dt.now()
    print_end_details(start_time, end_time)
    print("Evalutation of best performing model:")
    print(trials.best_trial['result']['loss'])

    # Dump trials object for safe-keeping
    with open(op.join(ckpt_dir, 'trials_{}.pkl'.format(start_time)), "wb") as pkl_file:
        pickle.dump(trials, pkl_file)
