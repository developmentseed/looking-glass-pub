"""
Apply model to SQS queue of messages containing overhead imagery URLs.

@author: developmentseed

pred_process_sqs.py
"""
import os.path as op
import csv
import requests
import itertools
import warnings

import numpy as np
from PIL import Image
import boto3
import tensorflow as tf
from keras import backend as K
from keras.utils import multi_gpu_model

from model_dlab import preprocess_input, BilinearUpsampling
from config import sqs_params as SP, pred_params as PP, ckpt_dir
from utils_training import load_model


overload_errors = ['<urlopen error [Errno 60] ETIMEDOUT>',
                   '<urlopen error [Errno 60] Operation timed out>',
                   '<urlopen error [Errno 2] Lookup timed out>',
                   '<urlopen error [Errno 8] Name or service not known>']

def get_messages_from_queue(queue, max_messages=10):
    """Generates messages from an SQS queue.

    Note: this continues to generate messages until the queue is empty.
    Every message on the queue will be deleted.
    """
    while True:
        resp = queue.receive_messages(MessageAttributeNames=['All'],
                                      MaxNumberOfMessages=max_messages)

        if resp:
            yield resp
        else:
            print('Queue is empty.')
            return


def download_images(messages, repeat_tries=10):
    """Download and return (as array) a single image"""

    img_arrs = []
    for message in messages:
        # Download the image
        while repeat_tries:
            url = message.body
            tile_ind = ''
            if message.message_attributes is not None:
                tile_ind = message.message_attributes.get('tile_info').get('StringValue')

            try:
                req = requests.get(url, stream=True)
                img_arr = np.array(Image.open(req.raw)).astype(np.float)
                req.close()
                img_arrs.append(img_arr)
                break

            except requests.exceptions.HTTPError as http_e:
                print('\nRecieved url error')
                #logging.error('HTTPError: %s', http_e)

            except requests.exceptions.InvalidURL as url_e:
                print('\nRecieved url error {}'.format(url_e))
                #logging.error('URL Error: %s', url_e)
                if str(url_e) in overload_errors:
                    print('Known load error, retrying')
                    #continue
                #else:
                #    return

            except Exception as err:
                print('\bRecieved other error')
                #logging.error('Other error on %s', url)

            repeat_tries -= 1
            if repeat_tries == 0:
                print('Too many repeats, quitting on {}'.format(tile_ind))
    return img_arrs


def process_queue(queue, model, csv_filepath, repeat_tries=10):
    """Calculate inference on a set of tiles"""

    with open(csv_filepath, 'a') as csv_file:
        writer = csv.writer(csv_file)

        sqm_per_pix = PP['img_res'] ** 2
        msg = 'Over-simplified area calculation. No equal area projection performed'
        warnings.warn(msg, Warning)

        # Loop through SQS messages
        for message_batch in get_messages_from_queue(queue):

            img_batch = download_images(message_batch)

            # If we get a valid image, predict
            if img_batch:
                # Add batch dimension if needed
                img_batch = np.array(img_batch).clip(0, 255)
                if len(img_batch.shape) == 3:
                    img_batch = img_batch[np.newaxis, ...]
                assert img_batch.ndim == 4, 'Size of image batch not correct'

                # Run prediction with model and get m^2
                y_p = model.predict(preprocess_input(img_batch), batch_size=10,
                                    verbose=0)
                masks = y_p[..., -1] >= PP['building_thresh']
                total_sq_m = sqm_per_pix * np.sum(masks, axis=tuple(range(1, masks.ndim)))

            # Write to file
            for sq_m, message in zip(total_sq_m, message_batch):

                if message.message_attributes is not None:
                    ti = message.message_attributes.get('tile_info').get('StringValue')
                    writer.writerow([ti, '{:.2f}'.format(sq_m)])

                    # Tell the queue that the message was processed
                    print('Processed message {}; {:.2f} m^2'.format(ti, sq_m))
                    message.delete()


if __name__ == "__main__":
    # Create SQS client
    sqs = boto3.resource('sqs', region_name=SP['region'])
    sqs_queue = sqs.get_queue_by_name(QueueName=SP['queue_name'])

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

    process_queue(sqs_queue, model, PP['csv_filepath'])
