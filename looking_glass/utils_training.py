"""
utils.py

@author: Development Seed

Utility functions for printing training details and saving/loading models
"""
import os
import os.path as op
import io
import shutil
import pprint

import numpy as np
from PIL import Image
import boto3
from botocore.exceptions import ClientError
import keras
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


from config import (ckpt_dir, tboard_dir, model_params as MP, save_params as SP,
                    img_aug_params as IMP)


def check_create_dir(dir_path):
    """Create a directory if it does not exist."""
    if not op.isdir(dir_path):
        os.mkdir(dir_path)


def print_start_details(start_time):
    """Print config at the start of a run."""
    pp = pprint.PrettyPrinter(indent=4)

    print('Start time: ' + start_time.strftime('%d/%m %H:%M:%S'))

    print('\nDatasets used:')
    pp.pprint(MP['training_data_dirs'])
    print('\nModel training details:')
    pp.pprint(MP)
    print('\nData Augmentation details:')
    pp.pprint(IMP)
    print('\n\n' + '=' * 40)


def print_end_details(start_time, end_time):
    """Print runtime information."""
    run_time = end_time - start_time
    hours, remainder = divmod(run_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print('\n\n' + '=' * 40)
    print('End time: ' + end_time.strftime('%d/%m %H:%M:%S'))
    print('Total runtime: %i:%02i:%02i' % (hours, minutes, seconds))


def copy_filenames_to_dir(file_list, dst_dir):
    """Copy a list of filenames (like images) to new directory."""
    for file_name in file_list:
        print('Copying: %s to %s' % (file_name, dst_dir))
        shutil.copy(file_name, dst_dir)

    print('Done.')


def save_model_yaml(model, model_fpath):
    """Save pre-trained Keras model."""
    with open(model_fpath, "w") as yaml_file:
        yaml_file.write(model.to_yaml())


def load_model(model_fpath, weights_fpath, custom_objects=None):
    """Load a model from yaml architecture and h5 weights."""
    from keras.models import model_from_yaml

    assert model_fpath[-5:] == '.yaml'
    assert weights_fpath[-3:] == '.h5'

    with open(model_fpath, "r") as yaml_file:
        yaml_architecture = yaml_file.read()

    model = model_from_yaml(yaml_architecture, custom_objects=custom_objects)
    model.load_weights(weights_fpath)

    return model


class TensorBoardImage(keras.callbacks.Callback):
    """Child class for creating tensorboard image summary callbacks"""

    def __init__(self, tag, tboard_save_dir, arr_imgs, img_type):
        super().__init__()
        self.tag = tag
        self.tboard_save_dir = tboard_save_dir
        self.arr_imgs = arr_imgs
        if img_type not in ['binary', 'sdt']:
            raise ValueError('`img_type` must be `binary` or `sdt`')
        self.img_type = img_type

    def on_epoch_end(self, epoch, logs={}):
        preds = self.model.predict(self.arr_imgs, batch_size=8)
        preds = preds.reshape((preds.shape[0], MP['input_shape'][0],
                               MP['input_shape'][1], preds.shape[-1]))
        preds = preds[:, :, :, -1]  # Grab only the building prediction channel

        if self.img_type == 'sdt':
            preds = plt.cm.bwr(preds)

        if np.max(preds) <= 1.0:  # Scale to 0-255 if necessary
            preds *= 255
            preds = preds.astype(np.uint8)

        writer = tf.summary.FileWriter(self.tboard_save_dir)

        img_summaries = []
        for pi, pred in enumerate(preds):

            image = Image.fromarray(pred)

            # Write the image to a string
            img_b = io.BytesIO()
            image.save(img_b, format='PNG')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=img_b.getvalue(),
                                       height=pred.shape[0],
                                       width=pred.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (self.tag, pi),
                                                  image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        writer.add_summary(summary, epoch)
        writer.close()

        return


def check_S3_existance(s3, bucket, key):
    """Check if key exists in an S3 bucket"""
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        return int(e.response['Error']['Code']) != 404
    return True


def sync_models_to_S3():
    """Helper to save models and tensorboard data to S3"""
    client = boto3.client('s3')

    # Save model architecture and parameters to S3
    for model_fname in os.listdir(ckpt_dir):
        rel_path = op.join(SP['sub_dir'], 'models', model_fname)

        if not check_S3_existance(client, SP['bucket_name'], rel_path):
            client.upload_file(op.join(ckpt_dir, model_fname),
                               SP['bucket_name'], rel_path)
            print('Uploaded {}'.format(op.join(SP['bucket_name'], rel_path)))


def sync_tb_dirs_to_S3():
    """Save tensorboard directories (containing training performance) to S3"""

    client = boto3.client('s3')

    # Requires some craftiness as boto3 doesn't support directory upload
    for root, _, filenames in os.walk(tboard_dir):
        for filename in filenames:

            # Get path on local disk and desired S3 path
            local_path = op.join(root, filename)
            s3_path = op.join(SP['sub_dir'], 'tensorboard',
                              op.relpath(local_path, tboard_dir))

            #print('local: {}; s3_path: {}'.format(local_path, s3_path))

            # If it doesn't exist on S3, upload it
            if not check_S3_existance(client, SP['bucket_name'], s3_path):
                client.upload_file(local_path, SP['bucket_name'], s3_path)
                print('Uploaded {}'.format(op.join(SP['bucket_name'],
                                                   s3_path)))
