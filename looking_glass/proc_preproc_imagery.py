"""
Preprocess imagery, create binary and SDT masks

@author: DevelopmentSeed

proc_preproc_imagery.py
"""
import os
import os.path as op

import numpy as np
from PIL import Image

from config import preproc_params as PP, data_dirs
from utils_labels import (preproc_sat_img, create_binary_mask, create_sdt_mask,
                          crop_directory)
from utils_training import check_create_dir


def run_preproc():
    """Preprocess raw satellite imagery and geojsons based on config params."""

    for raster_dir, raster_save_dir, label_binary_dir, label_sdt_dir, geojson_dir \
        in zip(PP['raster_dirs'],
               PP['raster_save_dirs'],
               PP['label_binary_dirs'],
               PP['label_sdt_dirs'],
               PP['geojson_dirs']):

        for a_dir in [raster_save_dir, label_binary_dir, label_sdt_dir, geojson_dir]:
            check_create_dir(a_dir)

        raster_fnames = [img for img in os.listdir(raster_dir)]
        geojson_fnames = [geoj for geoj in os.listdir(geojson_dir)]
        print('Processing {} images.'.format(len(raster_fnames)))

        ########################################
        # Load images, process, and create masks
        ########################################
        for fname_img, fname_geojson in zip(raster_fnames, geojson_fnames):
            # Check to make sure indices are aligned
            img_substr_ind = fname_img.index('img')
            geoj_substr_ind = fname_geojson.index('img')
            assert(op.splitext(fname_img)[0][img_substr_ind:] ==
                   op.splitext(fname_geojson)[0][geoj_substr_ind:])
            file_stem = op.splitext(fname_img)[0]

            # Preprocess raw satellite data and generate RGB rasters
            fname_raster = op.join(raster_dir, '{}.tif'.format(file_stem))
            fname_rgb_out = op.join(raster_save_dir, '{}.png'.format(file_stem))
            saved_rgb = preproc_sat_img(fname_raster, fname_rgb_out,
                                        zero_shift=True)
            # If RGB image wasn't saved (e.g., for many black pixels), continue
            if not saved_rgb:
                print('Skipped {} for too many black pixels'.format(fname_raster))
                continue

            # Create and save mask image
            fname_mask_out = op.join(label_binary_dir, '{}.png'.format(file_stem))
            fname_geojson = op.join(geojson_dir, fname_geojson)

            create_binary_mask(fname_geojson, fname_raster, fname_mask_out,
                               burn_val=1, no_data_val=0)

            # Create and save SDT image
            fname_sdt_out = op.join(label_sdt_dir, '{}.tif'.format(file_stem))
            create_sdt_mask(fname_mask_out, fname_sdt_out, PP['img_res'],
                            clip_vals=PP['sdt_clip'])

        ##################
        # Crop directories
        ##################
        print('\nMasks created, cropping.')
        targ_raster_crop_dir = raster_save_dir.replace('_Train/', '_Train_crop/')
        targ_bin_crop_dir = label_binary_dir.replace('_Train/', '_Train_crop/')
        targ_sdt_crop_dir = label_sdt_dir.replace('_Train/', '_Train_crop/')

        crop_directory(raster_save_dir, targ_raster_crop_dir,
                       target_size=PP['img_shape'][0:2])
        print('Finished cropping rasters')
        crop_directory(label_binary_dir, targ_bin_crop_dir,
                       target_size=PP['img_shape'][0:2])
        print('Finished cropping binary labels')
        crop_directory(label_sdt_dir, targ_sdt_crop_dir,
                       target_size=PP['img_shape'][0:2])
        print('Finished cropping signed distance transform')


def package_dir(rgb_dir, dict_dirs, savename_npz, img_shape):
    """Package images into npz file for ML training"""

    samp_fnames = [img for img in os.listdir(rgb_dir)]

    # Create dict of arrays that contain training and testing data
    dict_arr = {}
    for key in dict_dirs.keys():
        # Mask data arrays
        if key == 'y_binary':
            dtype = np.bool
        elif key == 'y_sdt':
            dtype = np.float16
        dict_arr[key] = np.empty((len(samp_fnames), img_shape[0], img_shape[1]),
                                 dtype=dtype)

    dict_dirs['x'] = rgb_dir
    dict_arr['x'] = np.empty((len(samp_fnames), img_shape[0], img_shape[1],
                              img_shape[2]), dtype=np.uint8)

    for si, samp_fname in enumerate(samp_fnames):
        for key, dir_path in dict_dirs.items():
            ext = '.tif' if 'sdt' in dir_path else '.png'

            fpath_img = op.splitext(op.join(dir_path, samp_fname))[0]
            dict_arr[key][si] = np.asarray(Image.open(fpath_img + ext))

    np.savez(savename_npz, **dict_arr)


if __name__ == '__main__':

    # Run RGB image and mask (binary/sdt) creation
    run_preproc()

    print('\nPackaging cropped data.')
    package_crops = True
    # Package data in ML-ready format
    for data_dir in data_dirs:
        if package_crops:
            data_dir += '_crop'
        package_dir(op.join(data_dir, 'RGB_png'),
                    dict_dirs=dict(y_binary=op.join(data_dir, 'label_binary'),
                                   y_sdt=op.join(data_dir, 'label_sdt')),
                    savename_npz=op.join(data_dir, 'data.npz'),
                    img_shape=PP['img_shape'])
