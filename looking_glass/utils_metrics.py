"""
utils_metrics.py

@author: Development Seed

Collection of evaluation metrics to use for evaluating image processing
"""
import numpy as np
from keras import backend as K
from keras.layers import multiply
from tensorflow import convert_to_tensor
import tensorflow as tf
from itertools import product

default_smooth = 1.
#weighting = config.model_params['class_weights']
weighting = [1., 6.667]



def _get_intersection(tensor_1, tensor_2):

    if len(tensor_1.shape) == 2 and len(tensor_2.shape) == 2:
        return K.sum(tensor_1 * tensor_2, axis=(1))
    elif len(tensor_1.shape) == 3 and len(tensor_2.shape) == 3:
        return K.sum(tensor_1 * tensor_2, axis=(1, 2))
    else:
        return K.sum(tensor_1 * tensor_2, axis=(1, 2, 3))


def _get_total_sum(tensor_1, tensor_2):

    if len(tensor_1.shape) == 2 and len(tensor_2.shape) == 2:
        return K.sum(tensor_1 + tensor_2, axis=(1))
    elif len(tensor_1.shape) == 3 and len(tensor_2.shape) == 3:
        return K.sum(tensor_1 + tensor_2, axis=(1, 2))
    else:
        return K.sum(tensor_1 + tensor_2, axis=(1, 2, 3))


def _get_total_square(tensor_1, tensor_2):
    if len(tensor_1.shape) == 2 and len(tensor_2.shape) == 2:
        return K.sum((K.square(tensor_1) + K.square(tensor_2)), axis=(1))
    elif len(tensor_1.shape) == 3 and len(tensor_2.shape) == 3:
        return K.sum((K.square(tensor_1) + K.square(tensor_2)), axis=(1, 2))
    else:
        return K.sum((K.square(tensor_1) + K.square(tensor_2)), axis=(1, 2, 3))


def get_f1_score1(y_true, y_pred, smooth=default_smooth):
    """Non-binary F1 Score (aka Sorenson-Dice) coeff. Useful for comparing
    images to labels

    Notes
    -----
    Definition here:
        https://en.wikipedia.org/wiki/F1_score
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    This mirrors what others in the ML community have been using where the pred
    and true vectors are summed in the denominator.
    """

    intersection = _get_intersection(y_true, y_pred)
    total = _get_total_sum(y_true, y_pred)

    #return K.mean((2. * intersection + smooth) / (total + smooth))
    return (2. * intersection + smooth) / (total + smooth)


def get_f1_score2(y_true, y_pred, smooth=default_smooth):
    """Get non-binary F1 Score (aka Sorenson-Dice) coeff. Useful for comparing
    images to labels

    Notes
    -----
    This mirrors what people *outside* the ML community seem to suggest for
    a non-binary F1 score by squaring the pred and true vectors.
    See "Comparison of nonbinary similarity coefficients for similarity
    searching, clustering and compound selection." by Al Khalifa (2009).
    """

    intersection = _get_intersection(y_true, y_pred)
    total = _get_total_square(y_true, y_pred)

    #return K.mean((2. * intersection + smooth) / (total + smooth))
    return (2. * intersection + smooth) / (total + smooth)


def get_f1_dist1(y_true, y_pred, smooth=default_smooth):
    """Helper to turn the F1 score into a loss"""
    return 1 - get_f1_score1(y_true, y_pred, smooth)


def get_f1_dist2(y_true, y_pred, smooth=default_smooth):
    """Helper to turn the F1 score into a loss"""
    return 1 - get_f1_score2(y_true, y_pred, smooth)


def get_jaccard_index1(y_true, y_pred, smooth=default_smooth):
    """Return Jaccard index for binary vectors.

    Note that the ML community seems to use this even when using non-binary
    vectors."""

    intersection = _get_intersection(y_true, y_pred)
    total = _get_total_sum(y_true, y_pred)

    return K.mean((intersection + smooth) / (total - intersection + smooth))


def get_jaccard_index2(y_true, y_pred, smooth=default_smooth):
    """Return Jaccard index for generalized for non-binary vectors.

    This mirrors what people *outside* the ML community seem to suggest for
    jaccard index with non-binary vectors. Theoretically, this seems superior.
    """

    intersection = _get_intersection(y_true, y_pred)
    total = _get_total_square(y_true, y_pred)

    return K.mean((intersection + smooth) / (total - intersection + smooth))


def get_jaccard_index1_from_sdt(y_true, y_pred, smooth=default_smooth):
    """Calculate jaccard index but from signed-dist vectors"""
    thresh = K.constant(0)
    y_true = K.cast(K.greater_equal(y_true, thresh), 'float32')
    y_pred = K.cast(K.greater_equal(y_pred, thresh), 'float32')

    ji = get_jaccard_index1(K.cast(K.argmax(y_true, axis=-1), 'int64'),
                            K.cast(K.argmax(y_pred, axis=-1), 'int64'),
                            smooth=K.constant(smooth))
    return ji


def get_jaccard_index1_from_sigmoid(y_true, y_pred, smooth=default_smooth):
    """Calculate jaccard index but from signed-dist vectors"""
    thresh = K.constant(0.5)
    ji = get_jaccard_index1(K.cast(K.greater_equal(y_true, thresh), 'float32'),
                            K.cast(K.greater_equal(y_pred, thresh), 'float32'),
                            smooth=K.constant(smooth))
    return ji

def get_jaccard_dist1(y_true, y_pred, smooth=default_smooth):
    """Helper to get Jaccard distance (for loss functions).

    Note: This mirrors what others in the ML community have been using even for
    non-binary vectors."""

    return 1 - get_jaccard_index1(y_true, y_pred, smooth)


def get_jaccard_dist2(y_true, y_pred, smooth=default_smooth):
    """Helper to get Jaccard distance (for loss functions).

    Note: This mirrors what people outside the ML community have been using."""

    return 1 - get_jaccard_index2(y_true, y_pred, smooth)


def get_weighted_mean_squared_error(y_true, y_pred, weight_buildings=True):
    """ """
    if weight_buildings:
        z_t = tf.constant(0, tf.float32)
        building_pixels = tf.cast(tf.greater_equal(y_true, z_t), tf.float32)
        weights = tf.add(tf.ones_like(y_true),
                         tf.multiply(building_pixels,
                                     tf.constant(weighting[1] - 1, tf.float32)))
        return K.mean(multiply([K.square(y_pred - y_true), weights]), axis=-1)

    return K.mean(K.square(y_pred - y_true), axis=-1)


def w_binary_crossentropy(y_true, y_pred, from_logits=False, weight_pos_pixels=True):
    """Helper to compute weights from y_true and compute loss

    Modified from Keras's tensorflow backend
    Parameters
    ----------
    y_true: A tensor with the same shape as `y_pred`.
    y_pred: A tensor.
    from_logits: Whether `y_pred` is expected to be a logits tensor.
        By default, we consider that `y_pred` encodes a probability
        distribution.

    Returns
    -------
        A tensor with the weighted binary crossentropy.
    """
    #y_true = K.squeeze(y_true, axis=-1)
    #y_pred = K.squeeze(y_pred, axis=-1)

    #if weight_pos_pixels is True:
    mult_val = K.constant(weighting[1] - 1)
    weights = K.ones_like(y_true, dtype='float32') + (y_true * mult_val)

    '''
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(K.common.epsilon(),
                                        y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))
    '''

    unweighted_loss = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    return K.mean(K.sum(weights * unweighted_loss, axis=-1))

    #return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weights)


def w_categorical_crossentropy(y_true, y_pred):
    """Not currently working"""
    raise RuntimeError('Loss function needs more work')

    weights = np.ones((2, 2))
    weights[0, 0] = weighting[0]
    weights[1, 1] = weighting[1]

    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, :, 0])  # 20, 0
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], K.shape(y_pred)[1], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (tf.cast(weights[c_t, c_p], tf.float32) *
                       tf.cast(y_pred_max_mat[:, :, c_p], tf.float32) *
                       tf.cast(y_true[:, :, c_t], tf.float32))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def get_lr_metric(optimizer):
    """Hack to get the optimizers learning rate for tensorboard"""

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
