"""
  Based on the into_record 
        __author__  = "casperneo@uchicago.edu"
  Modify post-process of hdf record correctly

  Main Usage:
  Read modis satellite image data from hdf files and write patches into tfrecords.
  Parallelized with mpi4py.
"""
__author__ = "tkurihana@uchicago.edu"

import tensorflow as tf
import os
import cv2
import json
import glob
import copy
import numpy as np

from mpi4py import MPI
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyhdf.SD import SD, SDC


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# FIXME: No longer used this function:gen_swaths
def gen_swaths(fnames, mode, resize):
    """Reads and yields resized swaths.
    Args:
        fnames: Iterable of filenames to read
        mode: {"mod09_tif", "mod02_1km"} determines wheter to processes the file as a tif
            or a hdf file
        resize: Float or None - factor to resize the image by e.g. 0.5 to halve height and
            width. If resize is none then no resizing is performed.
    Yields:
        filename, (resized) swath
    """

    # Define helper function to catch the exception from gdal directly
    def gdOpen(file):
        # print('Filename being opened by gdal:',file, flush=True)
        try:
            output = gdal.Open(file).ReadAsArray()
        except IOError:
            print("Error while opening file:", file, flush=True)
        return output

    if mode == "mod09_tif":
        # read = lambda tif_file: gdal.Open(tif_file).ReadAsArray()
        read = lambda tif_file: gdOpen(tif_file)

    elif mode == "mod02_1km":
        names_1km = {
            "EV_250_Aggr1km_RefSB": [0, 1],
            "EV_500_Aggr1km_RefSB": [0, 1],
            "EV_1KM_RefSB": [x for x in range(15) if x not in (12, 14)],
            # 6,7 are very noisy water vapor channels
            "EV_1KM_Emissive": [0, 1, 2, 3, 10, 11],
        }
        read = lambda hdf_file: read_hdf(hdf_file, names_1km)

    else:
        raise ValueError("Invalid reader mode", mode)

    rank = MPI.COMM_WORLD.Get_rank()
    for t in fnames:
        print("rank", rank, "reading", t, flush=True)

        try:
            swath = np.rollaxis(read(t), 0, 3)
        except Exception as e:
            print(rank, "Could not read", t, "because", e)
            continue

        if resize is not None:
            swath = cv2.resize(
                swath, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA
            )
        yield t, swath


def read_hdf(hdf_file, varname='varname'):
    """Read `hdf_file` and extract relevant varname .
    """
    hdf = SD(hdf_file, SDC.READ)
    return hdf.select(varname)

def proc_sds(sds_array):
    """
    IN: array = hdf_data.select(variable_name)
    """
    array = sds_array.get()
    array = array.astype(np.float32)
    
    # check bandinfo
    _bands = sds_array.attributes()['band_names']
    #print("Process bands", _bands)
    bands = _bands.split(",")
    
    # nan process
    nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
    if len(nan_idx) > 0:
        array[nan_idx] = np.nan
    else:
        pass
    #TODO in future, offse and scale argument also should be *kargs
    # radiacne offset
    offset = sds_array.attributes()['radiance_offsets']
    offset_array = np.zeros(array.shape) # new matrix
    offset_ones  = np.ones(array.shape)  # 1 Matrix 
    for iband in range(len(offset)):
        offset_array[iband, :,:] = array[iband, :,:] - offset[iband]*offset_ones[iband,:,:]
    
    # radiance scale
    scales = sds_array.attributes()['radiance_scales']
    scales_array = np.zeros(array.shape) # new matrix
    for iband in range(len(scales)):
        scales_array[iband, :,:] = scales[iband]*offset_array[iband,:,:]
    return scales_array, bands

def aug_array(ref_array, ems_array, ref_bands=[], ems_bands=[], cref_bands=[], cems_bands=[]):

    _,nx,ny = ref_array.shape

    # ref SB
    array_list = []
    for idx, iband in enumerate(cref_bands):
      for iref_band in ref_bands:
        if iband ==  iref_band:
          array_list+=[ref_array[idx].reshape(nx,ny,1)]
    
    # emissive SB
    for idx, iband in enumerate(cems_bands):
      for iems_band in ems_bands:
        if iband ==  iems_band:
          array_list+=[ems_array[idx].reshape(nx,ny,1)]
    # concatenation
    return np.concatenate(array_list, axis=2)

def gen_sds(filelist=[], ref_var='EV_500_Aggr1km_RefSB', ems_var='EV_1KM_Emissive',
            ref_bands=[], ems_bands=[] ):
  
  for ifile in filelist:
    ref_sds = read_hdf(ifile, varname=ref_var)
    ems_sds = read_hdf(ifile, varname=ems_var)
    ref_array, cref_bands = proc_sds(ref_sds)
    ems_array, cems_bands = proc_sds(ems_sds)

    # data concatenation
    swath = aug_array(ref_array, ems_array, 
                      ref_bands=ref_bands, ems_bands=ems_bands,
                      cref_bands=cref_bands, cems_bands=cems_bands)
    yield ifile, swath


def old_gen_patches(swaths, shape, strides):
    """Normalizes swaths and yields patches of size `shape` every `strides` pixels
    Args:
        swaths: Iterable of (filename, np.ndarray) to slice patches from
        shape: (height, width) patch size
        strides: (x_steps, y_steps) how many pixels between patches
    Yields:
        (filename, coordinate, patch): where the coordinate is the pixel coordinate of the
        patch inside of filename. BUG: pixel coorindate is miscalculated if swath is
        resized. Patches come from the swath in random order and are whiten-normalized.
    """
    stride_x, stride_y = strides
    shape_x, shape_y = shape

    for fname, swath in swaths:
        # NOTE: Normalizing the whole (sometimes 8gb) swath will double memory usage
        # by casting it from int16 to float32. Instead normalize and cast patches.
        # TODO other kinds of normalization e.g. max scaling.
        mean = swath.mean(axis=(0, 1)).astype(np.float32)
        std = swath.std(axis=(0, 1)).astype(np.float32)
        max_x, max_y, _ = swath.shape

        # Shuffle patches
        coords = []
        for x in range(0, max_x, stride_x):
            for y in range(0, max_y, stride_y):
                if x + shape_x < max_x and y + shape_y < max_y:
                    coords.append((x, y))
        np.random.shuffle(coords)

        for x, y in coords:
            patch = swath[x : x + shape_x, y : y + shape_y]
            # Filter away patches with Nans or if every channel is over 50% 1 value
            # Ie low cloud fraction.
            threshold = shape_x * shape_y * 0.5
            max_uniq = lambda c: max(np.unique(patch[:, :, c], return_counts=True)[1])
            has_clouds = any(max_uniq(c) < threshold for c in range(patch.shape[-1]))
            if has_clouds:
                patch = (patch.astype(np.float32) - mean) / std
                if not np.isnan(patch).any():
                    yield fname, (x, y), patch

def gen_patches(swaths, stride=64, patch_size=128, 
                 normalization=False, flag_nan=True, flag_shuffle=True):
    
    """Normalizes swaths and yields patches of size `shape` every `strides` pixels
        IN:  swath;   image data in hdf file
             stride;  # patch stride size. size/2 is defualt
             size;    # patch in x/y direction  
             normalization; z-score normalization. For MODIS data, MUST False
             flag_nan; # detect option for np.nan information
             flag_shuffle; shuffle data or not

        * Document 
        For Decoded SDS MODIS Dataset, User should NOT normalize the input data
        because z-score normalization mess up radiance infromation

        IF user get patches WITH Normalization
         normalization = True
         Otherwise, vals in patch are NOT normalized
        IF user want to get patches WITHOUT NAN value
         flag_nan=True
         Then, nanvalue will be excluded

    * Generic document by Casper
    Args:
        swaths: Iterable of (filename, np.ndarray) to slice patches from
        shape: (height, width) patch size
        strides: (x_steps, y_steps) how many pixels between patches
    Yields:
        (filename, coordinate, patch): where the coordinate is the pixel coordinate of the
        patch inside of filename. BUG: pixel coorindate is miscalculated if swath is
        resized. Patches come from the swath in random order and are whiten-normalized.
    """
    # params
    for fname, swath in swaths:
      # Numpy nan option
      if flag_nan:
        swath_mean = np.nanmean(swath, axis=(0,1))
        swath_std = np.nanstd(swath, axis=(0,1))
      else :
        swath_mean = swath.mean(axis=(0,1))
        swath_std = swath.std(axis=(0,1))
      # modify small std value 
      ill_stds = np.where(swath_std < 1.0e-20)[0]
      if len(ill_stds) > 0 :
        for idx in ill_stds:
          swath_std[idx] += 1.0e-20

      print(fname)
      print(swath.shape)
      max_x, max_y, _ = swath.shape

      # Shuffle patches
      coords = []
      for x in range(0, max_x, stride):
         for y in range(0, max_y, stride):
           if x + patch_size < max_x and y + patch_size < max_y:
              coords.append((x, y))
      if flag_shuffle:
        np.random.shuffle(coords)

      for i, j in coords:
        patch = swath[i:i + patch_size, j:j + patch_size]
        if normalization:
          patch -= swath_mean
          patch /= swath_std
        
        if not np.isnan(patch).any():
          yield fname, (i, j), patch
            

def write_feature(writer, filename, coord, patch):
    feature = {
        "filename": _bytes_feature(bytes(filename, encoding="utf-8")),
        "coordinate": _int64_feature(coord),
        "shape": _int64_feature(patch.shape),
        "patch": _bytes_feature(patch.ravel().tobytes()),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def get_blob_ratio(patch, thres_val=0.00):
    """ Compute Ratio of non-negative pixels in an image
        thres_val : threshold vale; defualt is 0/non-negative value
    """
    img = copy.deepcopy(patch[:,:,0]).flatten()
    clouds_ratio = len(np.argwhere(img > thres_val))/len(img)*100
    return clouds_ratio
    

def write_patches(patches, out_dir, patches_per_record):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
    Args:
        patches: Iterable of (filename, coordinate, patch) which defines tfrecord example
            to write.
        out_dir: Directory to save tfrecords.
        patches_per_record: Number of examples to save in each tfrecord.
    Side Effect:
        Examples are written to `out_dir`. File format is `out_dir`/`rank`-`k`.tfrecord
        where k means its the "k^th" record that `rank` has written.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    for i, patch in enumerate(patches):
        if i % patches_per_record == 0:
            rec = "{}-{}.tfrecord".format(rank, i // patches_per_record)
            print("Writing to", rec, flush=True)
            f = tf.python_io.TFRecordWriter(os.path.join(out_dir, rec))

        write_feature(f, *patch)

        print("Rank", rank, "wrote", i + 1, "patches", flush=True)


def get_args(verbose=False):
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument("source_glob", help="Glob of files to convert to tfrecord")
    p.add_argument("out_dir", help="Directory to save results")
    #FIXME here 
    # NOT USED NOW
    #p.add_argument(
    #    "mode",
    #    choices=["mod09_tif", "mod02_1km"],
    #    help="`mod09_tif`: Turn whole .tif swath into tfrecord. "
    #    "`mod02_1km` : Extracts EV_250_Aggr1km_RefSB, EV_500_Aggr1km_RefSB, "
    #    "EV_1KM_RefSB, and EV_1KM_Emissive.",
    #)
    p.add_argument(
        "--shape",
        type=int,
        help="patch size. Assume Square image",
        default=128,
    )
    # NOT USED NOW
    p.add_argument(
        "--resize",
        type=float,
        help="Resize fraction e.g. 0.25 to quarter scale. Only used for pptif",
    )
    p.add_argument(
        "--stride",
        type=int,
        help="patch stride. patch size/2 to compesate boundry information",
        default=64,
    )
    p.add_argument(
        "--ems_band",
        type=str,
        help="Emissive(Thermal/NearIR) additional band name. i.e. 28 or 29 or 31  ",
        default='29',
    )
    p.add_argument(
        "--patches_per_record", type=int, help="Only used for pptif", default=500
    )
    # NOT USED NOW
    #p.add_argument(
    #    "--interactive_categories",
    #    nargs="+",
    #    metavar="c",
    #    help="Categories for manually labeling patches. 'Noise' category will be added "
    #    "and those patches thrown away automatically.",
    #)

    FLAGS = p.parse_args()
    if verbose:
        for f in FLAGS.__dict__:
            print("\t", f, (25 - len(f)) * " ", FLAGS.__dict__[f])
        print("\n")

    FLAGS.out_dir = os.path.abspath(FLAGS.out_dir)
    return FLAGS


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    FLAGS = get_args(verbose=rank == 0)
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    fnames = []
    for i, f in enumerate(sorted(glob.glob(FLAGS.source_glob))):
        if i % size == rank:
            fnames.append(os.path.abspath(f))

    if not fnames:
        raise ValueError("source_glob does not match any files")

    #TODO make arg for ref_var & ems_var if using other modis dataset
    #TODO modify arg to add multiple temp/altitude bands  
    swaths  = gen_sds(fnames, 
                      ref_var='EV_500_Aggr1km_RefSB', ems_var='EV_1KM_Emissive',
                      ref_bands=["6","7"], ems_bands=["20"]+[FLAGS.ems_band])
    patches = gen_patches(swaths, FLAGS.stride, FLAGS.shape, 
                          normalization=False, flag_nan=True, flag_shuffle=True)

    write_patches(patches, FLAGS.out_dir, FLAGS.patches_per_record)

    print("Rank %d done." % rank, flush=True)
