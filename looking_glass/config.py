"""
Parameters for running tools in this repo.

config.py
"""

import os
import os.path as op
from hyperopt import hp

#import utils_metrics
from utils_metrics import (get_jaccard_index1, get_jaccard_index2,
                           get_jaccard_dist1, get_jaccard_dist2,
                           get_weighted_mean_squared_error,
                           get_jaccard_index1_from_sdt,
                           get_jaccard_index1_from_sigmoid,
                           w_categorical_crossentropy,
                           w_binary_crossentropy)
#from model_dlab import preprocess_input

# Set directories for saving model weights and tensorboard information
if os.environ['USER'] == 'ec2-user':
    ckpt_dir = op.join('/mnt', 'models')
    tboard_dir = op.join('/mnt', 'tensorboard')
    preds_dir = op.join('/mnt', 'preds')
    cloud_comp = True
else:
    ckpt_dir = op.join(os.environ['BUILDS_DIR'], 'looking-glass', 'models')
    tboard_dir = op.join(os.environ['BUILDS_DIR'], 'looking-glass', 'tensorboard')
    preds_dir = op.join(os.environ['BUILDS_DIR'], 'looking-glass', 'preds')
    plot_dir = op.join(os.environ['BUILDS_DIR'], 'looking-glass', 'plots')
    cloud_comp = False

if not op.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
if not op.isdir(tboard_dir):
    os.mkdir(tboard_dir)


data_folds_to_proc = ['AOI_5_Khartoum_Train',
                      'AOI_4_Shanghai_Train',
                      'AOI_2_Vegas_Train',
                      'AOI_3_Paris_Train']
data_dirs = [op.join(os.environ['DATA_DIR'], 'spacenet', temp_dir)
             for temp_dir in data_folds_to_proc]

preproc_params = dict(raster_dirs=[op.join(data_dir, 'RGB-PanSharpen')
                                   for data_dir in data_dirs],
                      raster_save_dirs=[op.join(data_dir, 'RGB_png')
                                        for data_dir in data_dirs],
                      label_binary_dirs=[op.join(data_dir, 'label_binary')
                                         for data_dir in data_dirs],
                      label_sdt_dirs=[op.join(data_dir, 'label_sdt')
                                      for data_dir in data_dirs],
                      geojson_dirs=[op.join(data_dir, 'geojson', 'buildings')
                                    for data_dir in data_dirs],
                      zero_shift=True,
                      max_rgb_val=255.,
                      img_res=0.31,  # m/pix
                      sdt_clip=(-128., 128.),
                      img_shape=(256, 256, 3))  # meters

model_params = dict(training_data_dirs=[trn_dir + '_crop' for trn_dir in data_dirs],
                    mask_type='sdt',  # 'binary' or 'sdt'
                    input_shape=(256, 256, 3),  # Image shape (HxWxC)
                    final_layer=[None, 'sigmoid', 'softmax', (-16., 16.)][1],  # Apply final act. to logits
                    num_classes=1,  # XXX: update to be prediction channels
                    n_gpus=1,
                    output_stride=16,  # 8 or 16
                    tf_preproc=True,  # Use tf_preproc to scale rgb images from -1 to 1

                    load_orig_weights=True,  # Whether to load Google's weights
                    orig_weights_fpath=op.join(ckpt_dir, 'deeplabv3_weights_tf_dim_ordering_tf_kernels.h5'),
                    optimizer_opts=[dict(opt_func='powersign',
                                         decay_func=[None, 'lin', 'cos', 'res'][0],
                                         decay_steps=5,
                                         decay_periods=2)],  # only for restart decay
                                    #dict(opt_func='adam')],
                                    #dict(opt_func='rmsprop')],
                    #loss=[w_categorical_crossentropy],
                    loss=[w_binary_crossentropy],
                    #loss=[get_weighted_mean_squared_error],

                    # (num epochs, freeze cutoff, learning rate)
                    phase_lengths=[(1, 'batch_norm_freeze 407', 1e-3),
                                   #(1, 'batch_norm', 1e-3),
                                   (1, 'batch_norm_freeze 358', 1e-3),  # End of exit flow (encoder)
                                   #(1, 'batch_norm', 1e-3),
                                   (2, 'batch_norm_freeze 291', 1e-3),  # Start of exit flow
                                   #(2, 195, 1e-3),  # Middle of middle flow
                                   #(1, 'batch_norm', 1e-3),
                                   #(2, 131, 1e-3),  # Start of middle flow
                                   (2, 99, 1e-3),  # Middle of entry flow
                                   #(1, 'batch_norm', 1e-4),
                                   (2, 67, 1e-3),  # End of entry flow
                                   (2, 7, 1e-4),  # Start of entry flow
                                   #(1, 'batch_norm', 5e-5),
                                   (2, 0, 1e-4),
                                   #(1, 'batch_norm', 1e-5),
                                   (2, 0, 1e-4),
                                   #(1, 'batch_norm', 1e-5),
                                   (2, 0, 1e-4),
                                   (2, 'batch_norm_freeze 358', 1e-4),
                                   (2, 'batch_norm_freeze 191', 1e-4),
                                   (2, 'batch_norm_freeze 0', 1e-4),
                                   (1, 'batch_norm', 1e-4),
                                   (3, 0, 1e-5)],

                    batch_size=20,  # Make as big as GPU can handle, should be >=16
                    #weight_init=['glorot_uniform'],
                    metrics=[get_jaccard_index1_from_sigmoid], #, get_jaccard_index2],
                    early_stopping_patience=10,  # Number of iters w/out val_acc increase
                    early_stopping_min_delta=0.01,
                    reduce_lr_patience=10,  # Number of iters w/out val_acc increase
                    reduce_lr_epsilon=0.02,
                    #class_weight={0: 1., 1: 6.667},  # XXX: Hard coded in utils_metrics
                    prop_total_img_set=1.0,  # Proportion of total images per train epoch
                    test_prop=0.1,  # test proportion,
                    max_sing_dataset_samps=4000,  # Max number of images from a single dataset
                    n_rand_hp_iters=3,
                    n_total_hp_iters=50,
                    shuffle_seed=42)  # Seed for random number generator

# Reset params for second main phase of training with OS=8
if False:
    model_params.update(phase_lengths=[(1, 99, 1e-4),  # Middle of entry flow
                                       (1, 'batch_norm', 1e-4),
                                       (2, 67, 1e-4),  # End of entry flow
                                       (1, 'batch_norm', 5e-5),
                                       (1, 67, 1e-4),  # End of entry flow
                                       (1, 'batch_norm', 5e-5),
                                       (2, 7, 1e-5),  # Start of entry flow
                                       (1, 'batch_norm', 5e-5),
                                       (2, 0, 1e-5),
                                       (1, 'batch_norm', 1e-5),
                                       (2, 0, 1e-5),
                                       (1, 'batch_norm', 1e-5),
                                       (2, 0, 1e-5),
                                       (2, 'batch_norm_freeze 358', 1e-5),
                                       (2, 'batch_norm_freeze 191', 1e-5),
                                       (2, 'batch_norm_freeze 0', 1e-5),
                                       (1, 'batch_norm', 1e-6),
                                       (3, 0, 1e-6)],
                        orig_weights_fpath=op.join(ckpt_dir, '0527_220222_MSE47.15_E23_weights.h5'),
                        output_stride=16,  # 8 or 16
                        batch_size=20)

# Debugging params for fast iteration
if not os.environ['USER'] == 'ec2-user':
    print('Local prediction: overriding number of images if running train_deeplab')
    model_params['prop_total_img_set'] = 0.01
    #model_params['test_prop'] = 0.005

img_aug_params = dict(horizontal_flip=True,
                      vertical_flip=True,
                      #rotation_range=90.,
                      #zoom_range=(0.8, 2.),
                      #shear_range=15.,
                      #channel_shift_range=5.,
                      fill_mode='reflect')

# Where to save trained models
save_params = dict(save_models_to_S3=True,
                   bucket_name='ds-ml-labs',
                   sub_dir='looking_glass_models')

sqs_params = dict(geojson_fpaths=[op.join(os.environ['BUILDS_DIR'], 'looking-glass', 'preds', 'bounds', 'detroit.geojson')],
                  region='us-east-1',
                  queue_name='looking-glass-sqs',
                  message_body='https://api.mapbox.com/v4/digitalglobe.2lnpeioh/{z}/{x}/{y}.jpg?access_token={token}',
                  zoom=18)

pred_params = dict(n_gpus=1,
                   #orig_weights_fpath=op.join(ckpt_dir, ),
                   model_weights_fname=op.join(ckpt_dir, '0524_215014_weights.h5'),
                   model_arch_fname=op.join(ckpt_dir, '0524_215014_arch.yaml'),
                   img_res=0.31,  # In meters
                   # Threshold [0-1] for building pixel when binarizing prediction. Should be:
                       # [0-1] for models trained on binary masks
                       # 0 for models trained on SDT masks
                   building_thresh=0.,
                   csv_filepath=op.join(preds_dir, 'pred_areas_detroit_model_0524_215014.csv'))

# XXX Plotting uses model weights/arch specified in `pred_params`
inference_plot_params = dict(#data_set_fpath=op.join(os.environ['DATA_DIR'], 'jakarta', 'data', 'data.npz'),
                             data_set_fpath=op.join(os.environ['DATA_DIR'], 'spacenet', 'AOI_5_Khartoum_Train_crop', 'data.npz'),
                             data_type='spacenet', #'label-maker',  # 'label-maker' or 'spacenet'
                             mask_type='sdt',  # 'binary' or 'sdt'
                             n_plots=147,
                             save_dir=op.join(preds_dir, 'example_plots'),
                             img_res=0.31)  # In meters
######################
# Params for hyperopt
######################
def get_params(MP):
    """Return hyperopt parameters"""

    return dict(
        optimizer=hp.choice('optimizer', MP['optimizer_opts']),
        #weight_init=hp.choice('weight_init', MP['weight_init']),
        loss=hp.choice('loss', MP['loss']))
