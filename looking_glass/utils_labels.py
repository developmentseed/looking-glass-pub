"""
utils_labels.py

Utilities for creating labeled data

Some code modified from avanetten:
    https://gist.github.com/avanetten
"""

import os
import os.path as op
import sys
import json
import numpy as np
import skfmm
from osgeo import gdal, osr, ogr
from PIL import Image
from utils_training import check_create_dir


def preproc_sat_img(fname_raster, savename_raster, zero_shift=True,
                    max_val=255., skip_threshold=0.9, percentile=98.):
    """Preprocess raw satellite imagery where band vals are outside 0-255.

    skip_threshold: float
        Skip images that have too many black pixels (without data).
    """

    img_arr = gdal.Open(fname_raster).ReadAsArray()
    img_arr = np.moveaxis(img_arr, 0, -1)  # Roll axes so channels is last

    # Zero the lower bound
    if zero_shift:
        img_arr = img_arr - np.min(img_arr, (0, 1))  # shift color range to 0

    # Set upper bound
    data_max_val = np.percentile(img_arr, percentile, axis=(0, 1))
    img_arr = img_arr / data_max_val * max_val  # Normalize to 0 - max_val

    # Clip between 0-255
    np.clip(img_arr, 0, max_val, img_arr)

    non_black_prop = np.sum(np.mean(img_arr, -1) > 0.) / (img_arr.shape[0] * img_arr.shape[1])

    # Save image
    if skip_threshold:
        if non_black_prop >= skip_threshold:
            dist_img = Image.fromarray(np.uint8(img_arr))  # Convert to 0-255 ints
            dist_img = dist_img.convert(mode='RGB')
            dist_img.save(savename_raster, compression=0)
            return True
    return False

def create_binary_mask(fname_geojson, fname_raster, savename_mask,
                       burn_val=255, no_data_val=0):
    """Generate and save binary mask given geojson and raster image.

    fname_geojson: str
        Filepath to geojson file with building masks
    fname_raster: str
        Filepath to GeoTIFF raster image
    """

    if op.splitext(fname_raster)[1] != '.tif':
        raise ValueError('`fname_raster` must point to a GeoTIFF')

    # Load geojson
    vector_file = ogr.Open(fname_geojson)
    vector_layer = vector_file.GetLayer()

    # Load the raster image (just to match for image size)
    raster = gdal.Open(fname_raster)
    sz_x, sz_y = raster.RasterXSize, raster.RasterYSize

    # Create the destination data source
    mask = gdal.GetDriverByName('GTiff').Create(savename_mask, sz_x, sz_y, 1,
                                                gdal.GDT_Byte)
    mask.SetGeoTransform(raster.GetGeoTransform())
    mask.SetProjection(raster.GetProjection())
    band = mask.GetRasterBand(1)

    # XXX: At least ipynb crashes when setting no-data val
    #band.SetNoDataValue(no_data_val)

    # Rasterize
    gdal.RasterizeLayer(mask, [1], vector_layer, burn_values=[burn_val])
    band.FlushCache()


def create_sdt_mask(fname_bin_mask, savename_SDT_mask, m_per_pixel,
                    clip_vals=(-128., 128.), act_func=None):
    """Generate and save signed-distance transform mask
    Parameters
    ----------
    fname_bin_mask: str
        Filepath to binary mask file
    savename_SDT_mask: str
        Save path for signed_dist transform mask
    m_per_pixel: float
        Number of meters per img pixel (i.e., spatial resolution)
    clip_vals: tuple
        Floats specifying how far away from buildings (in meters) before SDT
        values will be clipped.
    act_func: function
        Should take and return a numpy array
    """

    # Load binary mask
    bin_mask = np.asarray(Image.open(fname_bin_mask))

    # Create mask using signed distance function
    fmm_arr = np.where(bin_mask >= 1, 1, -1)  # Set all values to 1 or -1

    # Check for no buildings
    if np.all(fmm_arr == -1):
        dist_arr = np.zeros_like(fmm_arr).astype(np.float)
        dist_arr.fill(clip_vals[0])
    elif np.all(fmm_arr == 1):
        dist_arr = np.zeros_like(fmm_arr).astype(np.float)
        dist_arr.fill(clip_vals[1])
    else:
        dist_arr = skfmm.distance(fmm_arr, dx=m_per_pixel).clip(clip_vals[0],
                                                                clip_vals[1])

    # Apply activation function if necessary
    if act_func:
        dist_arr = act_func(dist_arr)

    # Save image out
    dist_img = Image.fromarray(dist_arr)
    dist_img = dist_img.convert(mode='F')
    dist_img.save(savename_SDT_mask)#, compression='tiff_ccitt')


def crop_directory(orig_dir, target_dir, target_size=(256, 256)):
    """Helper to crop a directory of images"""

    # Make sure save directory exists
    check_create_dir(target_dir)

    image_fnames = os.listdir(orig_dir)

    # Loop over all images in the original directory
    for img_fname in image_fnames:
        img = Image.open(op.join(orig_dir, img_fname))
        img.load()

        width, height = img.size

        row_starts = range(0, height, target_size[0])
        col_starts = range(0, width, target_size[1])
        n_crops = 0

        # Loop over crops
        for c_start in col_starts[:-1]:
            for r_start in row_starts[:-1]:
                # bbox is (left, upper, right, lower)
                crop_img = img.crop((c_start, r_start,
                                     c_start + target_size[1],
                                     r_start + target_size[0]))
                save_name = '{}_crop_{}{}'.format(op.splitext(img_fname)[0],
                                                  n_crops,
                                                  op.splitext(img_fname)[1])
                #non_black_pix = np.sum(np.mean(np.array(crop_img), -1) > 0)
                # Save cropped image
                crop_img.save(op.join(target_dir, save_name))
                n_crops += 1


def _latLonToPixel(lat, lon, input_raster, targetsr="", geom_transform=""):
    """Convert geospatial point to pixel.

    Modified from SpaceNetUtilities
    """

    # Get spatial reference frame
    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(lon, lat)

    if targetsr == "":
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)

    if geom_transform == "":
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    y_origin = transform[3]

    pixel_width = transform[1]
    pixel_height = transform[5]

    geom.Transform(coord_trans)
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)


def geojson_to_pixel_arr(raster_file, geojson_file, pixel_ints=True,
                         verbose=False):
    """
    Tranform geojson file into array of points in pixel (and latlon) coords

    Parameters:
    -----------
    raster_file: str
        Filepath to raster image
    geojson_file: str
        Filepath to geojson file
    pixel_ints: bool
    verbose: str

    Returns:
    --------
    pixel_coords: list
    latlon_coords: list
    """

    ######################
    # Load relevant files
    ######################
    with open(geojson_file) as geoj_f:
        geojson_data = json.load(geoj_f)

    # load raster file and get geo transforms
    src_raster = gdal.Open(raster_file)
    targetsr = osr.SpatialReference()
    targetsr.ImportFromWkt(src_raster.GetProjectionRef())

    geom_transform = src_raster.GetGeoTransform()

    ########################
    # Process geo information
    ########################
    # get latlon coords
    latlons = []
    types = []
    for feature in geojson_data['features']:
        coords_tmp = feature['geometry']['coordinates'][0]
        type_tmp = feature['geometry']['type']
        if verbose:
            print("features: {}".format(feature.keys()))
            print("geometry features: {}".format(feature['geometry'].keys()))

        latlons.append(coords_tmp)
        types.append(type_tmp)

    # convert latlons to pixel coords
    pixel_coords = []
    latlon_coords = []
    # TODO: refactor for DRY
    for poly_type, poly0 in zip(types, latlons):

        if poly_type.upper() == 'MULTIPOLYGON':
            for poly in poly0:
                poly = np.array(poly)
                if verbose:
                    print("poly: {}".format(poly))
                    print("poly.shape: {}".format(poly.shape))

                # account for nested arrays
                if len(poly.shape) == 3 and poly.shape[0] == 1:
                    poly = poly[0]

                poly_list_pix = []
                poly_list_latlon = []

                for coord in poly:
                    if verbose:
                        print('coord: {}'.format(coord))
                    lon, lat, z = coord
                    px, py = _latLonToPixel(lat, lon, input_raster=src_raster,
                                            targetsr=targetsr, geom_transform=geom_transform)
                    poly_list_pix.append([px, py])
                    if verbose:
                        print("px, py: {} {}".format(px, py))
                    poly_list_latlon.append([lat, lon])

                if pixel_ints:
                    ptmp = np.rint(poly_list_pix).astype(int)
                else:
                    ptmp = poly_list_pix
                pixel_coords.append(ptmp)
                latlon_coords.append(poly_list_latlon)

        elif poly_type.upper() == 'POLYGON':
            poly = np.array(poly0)

            # account for nested arrays
            if len(poly.shape) == 3 and poly.shape[0] == 1:
                poly = poly[0]

            poly_list_pix = []
            poly_list_latlon = []

            for coord in poly:
                if verbose:
                    print('coord: {}'.format(coord))
                lon, lat, z = coord
                px, py = _latLonToPixel(lat, lon, input_raster=src_raster,
                                        targetsr=targetsr, geom_transform=geom_transform)
                poly_list_pix.append([px, py])
                if verbose:
                    print("px, py: {} {}".format(px, py))
                poly_list_latlon.append([lat, lon])

            if pixel_ints:
                ptmp = np.rint(poly_list_pix).astype(int)
            else:
                ptmp = poly_list_pix
            pixel_coords.append(ptmp)
            latlon_coords.append(poly_list_latlon)

        else:
            raise ValueError("Unknown shape type in coords_arr_from_geojson()")

    return pixel_coords, latlon_coords
