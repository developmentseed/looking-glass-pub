"""
Functionality to build DeeplabV3+ in Keras

Modified from bonlime's implementation:
    https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py

model_dlab.py
"""

import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import (Input, BatchNormalization, Activation, Conv2D,
                          Concatenate, Reshape, multiply, Lambda)
from keras.layers import DepthwiseConv2D, ZeroPadding2D
from keras.layers import AveragePooling2D, Dropout
from keras import backend as K
from keras.applications import imagenet_utils

from keras.engine import Layer, InputSpec
from keras.utils import conv_utils


class BilinearUpsampling(Layer):
    """Bilinear upsampling layer"""

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None,
                 **kwargs):
        """Initialize bilinearupsamping layer

        Parameters:
        ----------
        upsampling: tuple
            2 numbers > 0. The upsampling ratio for h and w
        output_size: int
            Used instead of upsampling arg if passed
        """

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1,
               depth_activation=False, epsilon=1e-3):
    """SepConv with BN between depthwise & pointwise. Optionally add activation after BN

    Implements right "same" padding for even kernel sizes

    Parameters:
    -----------
    x: input tensor
    filters: num of filters in pointwise convolution
    prefix: prefix before name
    stride: stride at depthwise conv
    kernel_size: kernel size for depthwise convolution
    rate: atrous rate for depthwise convolution
    depth_activation: flag to use activation between depthwise & poinwise convs
    epsilon: epsilon to use in BN layer

    Returns
    -------
    x: tensor
        Output tensor with Seperable conv and BatchNorm applied
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride),
                        dilation_rate=(rate, rate), padding=depth_padding,
                        use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)

    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)

    if depth_activation:
        x = Activation('relu')(x)

    return x


def conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes.

    Without this there is a 1 pixel drift when stride = 2

    Parameters:
    -----------
    x: input tensor
    filters: num of filters in pointwise convolution
    prefix: prefix before name
    stride: stride at depthwise conv
    kernel_size: kernel size for depthwise convolution
    rate: atrous/dilation rate for depthwise convolution

    Returns:
    --------
    x: tensor
        Output tensor with dilated Conv2D applied (according to params)
    """

    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)

        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                   rate=1, depth_activation=False, return_skip=False):
    """
    Module of XCeption block.

    Parameters:
    -----------
    inputs: input tensor
    depth_list: number of filters in each SepConv layer. len(depth_list) == 3
    prefix: prefix before name
    skip_connection_type: one of {'conv','sum','none'}
    stride: stride at depthwise conv
    rate: atrous rate for depthwise convolution
    depth_activation: flag to use activation between depthwise & poinwise convs
    return_skip: flag to return additional tensor after 2 SepConvs for decoder

    Returns:
    --------
    output: tensor
        Computation block for applying XCeption computation block
    """

    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                               kernel_size=1,
                               stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def get_deepLabV3p(input_shape, num_classes=21, final_layer='sigmoid',
                   output_stride=16):
    """
    DeepLab V3+ model.

    Parameters:
    ----------
        input_shape: iterable
            Shape of input image. format HxWxC.
        num_classes: int
            Number of desired classes. If != 21 (as in orig paper), last layer
            is initialized randomly.
        final_layer: str or 2-tuple
            Flag to for final activation layer for model's logits. If tuple,
            will use tanh and scale/shift to tuple's range
        output_stride: int
            Output stride, which determines input_shape/feature_extractor_output
            ratio. One of {8,16}.
    """

    if output_stride == 8:
        entry_block3_stride = 1
        middle_block_rate = 2 # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    elif output_stride == 16:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)
    else:
        raise ValueError('`OS` must be 8 or 16. Received: {}'.format(output_stride))

    # Set image size
    img_input = Input(input_shape)

    # Initial convolutional layers
    x = Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1',
               use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    x = conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    # First XCeption block
    x = xception_block(x, [128, 128, 128], 'entry_flow_block1',
                       skip_connection_type='conv', stride=2,
                       depth_activation=False)
    x, skip1 = xception_block(x, [256, 256, 256], 'entry_flow_block2',
                              skip_connection_type='conv', stride=2,
                              depth_activation=False, return_skip=True)

    x = xception_block(x, [728, 728, 728], 'entry_flow_block3',
                       skip_connection_type='conv', stride=entry_block3_stride,
                       depth_activation=False)
    # Add middle xception blocks
    for i in range(16):
        x = xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                           skip_connection_type='sum', stride=1,
                           rate=middle_block_rate, depth_activation=False)

    x = xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                       skip_connection_type='conv', stride=1,
                       rate=exit_block_rates[0], depth_activation=False)
    x = xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                       skip_connection_type='none', stride=1,
                       rate=exit_block_rates[1], depth_activation=True)

    # End of feature extractor
    # Branching for Atrous Spatial Pyramid Pooling

    # Simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # Rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # Hole = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # Hole = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # Image Feature branch
    out_shape = int(np.ceil(input_shape[0] / output_stride))
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    # Concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    ######################
    # DeepLab v.3+ decoder
    ######################

    # Feature projection
    # x4 (x2) block
    x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                        int(np.ceil(input_shape[1] / 4))))(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    x = Conv2D(num_classes, (1, 1), padding='same', name='final_logits')(x)
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)

    if final_layer is None:
        pass
    elif final_layer == 'sigmoid':
        x = Activation('sigmoid')(x)
    elif final_layer == 'softmax':
        x = Activation('softmax')(x)
    elif isinstance(final_layer, tuple):
        x = Activation('tanh')(x)
        # Modify tanh bounds to match the min/max value of the SDT data
        zero_pt = (final_layer[1] + final_layer[0]) / 2
        output_range = (final_layer[1] - final_layer[0]) / 2
        x = Lambda(lambda z: z * output_range + zero_pt)(x)
    else:
        raise ValueError('Got unknown final layer type')

    x = Reshape((input_shape[0] * input_shape[1], num_classes))(x)

    model = Model(img_input, x, name='DeepLabV3p')

    return model


def preprocess_input(x, copy=False):
    """Preprocesses a numpy array encoding a batch of images.

    Parameters:
    ----------
        x: ndarray
            4D numpy array consists of RGB values within [0, 255].
        copy: bool
            Whether or not to copy the array before running preprocessing

    Returns:
    -------
        x: Preprocessed array scaled from -1 to 1 sample-wise.
    """

    if copy:
        return imagenet_utils.preprocess_input(x.copy(), mode='tf')
    else:
        return imagenet_utils.preprocess_input(x, mode='tf')
