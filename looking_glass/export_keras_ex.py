"""
Script demonstrating how to export a Keras model. A version of the model
saved from Keras should be on disk locally.

@author: Development Seed
"""

import os.path as op
from shutil import copy2
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.estimator import model_to_estimator
from tensorflow.contrib.distributions import percentile
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.utils import conv_utils

from config_model import (model_start_time, model_dir, export_dir, IMG_SIZE,
                          PCT_CUTOFF, N_GPU)


def serving_input_receiver_fn():
    """Convert string encoded images into preprocessed tensors.

    Baked into the TF Estimator object when prepping to make a TF Serving image
    """

    def decode_and_resize(image_str_tensor):
        """Decodes an image string, resizes it, and performs custom equalization."""
        image = tf.image.decode_image(image_str_tensor, channels=IMG_SIZE[2])
        image = tf.reshape(image, IMG_SIZE)

        # Shift bottom of color range to 0
        image = image - tf.reduce_min(image, axis=(0, 1))

        # Divide pixel intensity by some portion of max value
        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, percentile(image, PCT_CUTOFF))

        return image

    # Run preprocessing for batch prediction
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    images_tensor = tf.map_fn(decode_and_resize, input_ph, back_prop=False,
                              dtype=tf.float32)
    images_tensor = tf.clip_by_value(images_tensor, 0., 1.)

    return tf.estimator.export.ServingInputReceiver(
        {'input_1': images_tensor},  # The key should match your model's first layer. Try `my_model.input_names`
        {'image_bytes': input_ph})   # You can specify the key here, but this is a good default


class BilinearUpsampling(Layer):
    """Custom bilinear upsampling layer.

    Note: If doing model development, this now exists as a default in Keras:
        https://keras.io/layers/convolutional/#upsampling2d
    """


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
        #self.data_format = K.normalize_data_format(data_format)
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
            return tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                     inputs.shape[2] * self.upsampling[1]),
                                            align_corners=True)
        else:
            return tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                     self.output_size[1]),
                                            align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_keras_model(arch_fpath, weights_fpath, custom_objects=None, n_gpus=0):
    """Load a Keras architecture and weights

    Parameters
    ----------
    arch_fpath: str
        Architecture saved as YAML file
    weights_fpath: str
        Weights saved as h5 file
    custom_objects: dicts
        Custom objects needed when loading the Keras model.
    n_gpus: int
        Number of gpus available to run prediction. Default 0.

    Returns
    -------
    model: keras.engine.training.Model
    """

    if not op.splitext(arch_fpath)[-1] == '.yaml':
        raise ValueError('Model filepath must have `.yaml` extension.')
    if not op.splitext(weights_fpath)[-1] == '.h5':
        raise ValueError('Weights filepath must have `.h5` extension.')

    with open(arch_fpath, "r") as yaml_file:
        yaml_architecture = yaml_file.read()

    if n_gpus > 1:
        # Load weights on CPU to avoid taking up GPU space
        with tf.device('/cpu:0'):
            template_model = model_from_yaml(yaml_architecture,
                                             custom_objects=custom_objects)
            template_model.load_weights(weights_fpath)

            for layer in template_model.layers:
                layer.trainable = False

        model = multi_gpu_model(template_model, gpus=n_gpus)
    # If on only 1 gpu (or cpu), train as normal
    else:
        model = model_from_yaml(yaml_architecture,
                                custom_objects=custom_objects)
        model.load_weights(weights_fpath)

        for layer in model.layers:
            layer.trainable = False

    return model


def export_model(fpath_arch, fpath_weights, export_dir, reciever_fn,
                 custom_objects=None, n_gpu=0):
    """Export a saved Keras model to a TF Estimator object

    Parameters
    ----------
    fpath_arch: str
        Filepath to model architecture YAML file.
    fpath_weights: str
        Filepath to model weights h5 file.
    export_dir: str
        Weights saved as h5 file
    reciever_fn: function
        Serving input receiver function for TF
    custom_objects: dicts
        Custom objects needed when loading the Keras model into TF.
        Default: None
    n_gpu: int
        Number of GPUs expected during inference. Default: 0

    Returns
    -------
    model: keras.engine.training.Model
    """

    print(f'Loading model with arch and weights:\n{fpath_arch}\n{fpath_weights}')
    model = load_keras_model(fpath_arch, fpath_weights, custom_objects_dict,
                             n_gpu)
    # Compile model (necessary for creating an estimator; no training done)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    print('Exporting model as estimator...')
    # Create an Estimator object, and save to disk with preprocessing function
    estimator_dir = op.join(export_dir, 'estimator')
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                      model_dir=estimator_dir,
                                                      custom_objects=custom_objects_dict)

    # TF adds a `keras` subdirectory (unsure why), so need to copy the
    #     estimator files up one level to the `estimator` directory
    estimator_keras_dir = Path(estimator_dir) / 'keras'
    for est_file in estimator_keras_dir.glob('*'):
        copy2(est_file, estimator_dir)

    # The below function will be renamed to `estimator.export_saved_model` in TF 2.0
    return_dir = estimator.export_savedmodel(
        export_dir, serving_input_receiver_fn=reciever_fn)
    print(f'Estimator saved. Use files in {return_dir} to make TF Serving image.')


if __name__ == '__main__':

    custom_objects_dict = {'BilinearUpsampling': BilinearUpsampling}
    fpath_arch = op.join(model_dir, f'{model_start_time}_arch.yaml')
    fpath_weights = op.join(model_dir, f'{model_start_time}_arch.yaml')

    export_model(fpath_arch, fpath_weights, export_dir,
                 serving_input_receiver_fn, custom_objects_dict,
                 N_GPU)

'''
# To save the model to a TF serving cnotainer, see the example here: https://www.tensorflow.org/serving/docker

# Copying from the example, modify as needed:

cd ${BUILDS_DIR}/looking-glass
docker run -d --name serving_base tensorflow/serving:1.13.0
docker cp looking_glass_export serving_base:/models/looking_glass
docker commit --change "ENV MODEL_NAME looking_glass" serving_base developmentseed/looking-glass:v2
docker kill serving_base
docker container prune

# To push to DevSeed's account on dockerhub
docker push developmentseed/looking-glass:v2

# See the example in `docker_pred_example.ipynb` to run inference with real images

# To run the GPU version
# See tensorflow's website for requirements:https://www.tensorflow.org/serving/docker#serving_with_docker_using_your_gpu
'''
