"""
  Based on the into_mod_record 
        __author__  = "tkurihana@uchicago.edu"
  Modify post-process of hdf record correctly

  Modify post-process of hdf to apply z-score normalization.
  Before running this computation, user has to compute global mean and deviation
  for each input bands.

  Main Usage:
  Read modis satellite image data from hdf files and write patches into tfrecords.
  Parallelized with mpi4py.
"""
__author__ = "tkurihana@uchicago.edu"

import tensorflow as tf
import os
import cv2
import sys
import glob
import copy
import numpy as np

from mpi4py   import MPI
from pathlib  import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyhdf.SD import SD, SDC

# own library
homedir=str(Path.home())
clouds_dir="/clouds/src_analysis/lib_hdfs"
sys.path.insert(1,os.path.join(sys.path[0],homedir+clouds_dir))
from alignment_lib import gen_mod35_img
from alignment_lib import get_filepath
from alignment_lib import translate_const_clouds_array

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def read_hdf(hdf_file, varname='varname'):
    """Read `hdf_file` and extract relevant varname .
    """
    hdf = SD(hdf_file, SDC.READ)
    return hdf.select(varname)

def proc_sds(sds_array, sdsmax=32767):
    """
    IN: array = hdf_data.select(variable_name)
    """
    array = sds_array.get()
    #array = array.astype(np.float32)
    array = array.astype(np.float64)
    
    # check bandinfo
    _bands = sds_array.attributes()['band_names']
    #print("Process bands", _bands)
    bands = _bands.split(",")
    
    # error code 
    # higher value than 32767
    # C6 MODIS02 version, 65500-65534 are error code defined by algorithm
    # 65535 is _FillValue 
    # So far 65535(FillValue) is the largest value in native 16bit integer
    err_idx = np.where( (array > sdsmax) 
                      & (array < sds_array.attributes()['_FillValue']) )
    # nan process
    nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
    if len(nan_idx) > 0:
        array[nan_idx] = np.nan
    else:
        pass
    if len(err_idx) > 0:
        array[err_idx] = np.nan
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
    # Abort stopper for HDF-open error    
    if ref_sds is None:
      # call mpiabort function
      sys.excepthook =  mpiabort_excepthook
      sys.excepthook = sys.__excepthook__  # 2nd call?
    #FIXME Probably this second if statement is not necessary.
    if ems_sds is None:
      # call mpiabort function
      sys.excepthook =  mpiabort_excepthook
      sys.excepthook = sys.__excepthook__  # 2nd call?

    ref_array, cref_bands = proc_sds(ref_sds)
    ems_array, cems_bands = proc_sds(ems_sds)

    # data concatenation
    swath = aug_array(ref_array, ems_array, 
                      ref_bands=ref_bands, ems_bands=ems_bands,
                      cref_bands=cref_bands, cems_bands=cems_bands)
    yield ifile, swath


def get_masks(rpatch_size, channels):

    # mask
    # making mask array same shape as resized array
    #mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float32)
    mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float64)
    cv2.circle(mask, center=(rpatch_size // 2, rpatch_size // 2), 
               radius=rpatch_size//2, color=1, thickness=-1)
    mask = np.expand_dims(mask, axis=-1)
    
    #  multiple dimension
    mask_list = [ mask for i in range(channels)]
    masks = np.concatenate(mask_list, axis=-1)     
    return masks


def gen_patches(swaths, stride=64, patch_size=128, rpatch_size=128, channels=6,
                 normalization=False, flag_shuffle=False,
                 global_mean=np.zeros((6)), global_stdv=np.ones((6)),
                 mod35_datadir='./', thres=0.3):
    
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

        UPDATE: 2019/07/25
          To normalize input correctly, user apply normalization scheme with
          computed global_mean and global_stdv
          For usage, turn normalization True and give global_mean and global_stdv

        UPDATE:2019/11/08
          To mitigate effect of rotation, padding and only consider circle of the center


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
    # checkio for normalizaion scheme
    if normalization:
        norm_ok_flag = 0
        # check initial array are replaced by parsed computed array
        if not global_mean.all() == 0.000:
          norm_ok_flag += 1
          if not global_stdv.all() == 1.000 :
            norm_ok_flag += 1
        try:
          if norm_ok_flag == 2:
             print(" Apply Normalization scheme by global mean&stdv ", flush=True)
        except:
          raise ValueError(" Correct global mean & stdv are not parsed")
    
    # mask
    # making mask array same shape as resized array
    #
    # Functionize lines below
    masks = get_masks(rpatch_size, channels)
    #mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float32)
    #mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float64)
    #cv2.circle(mask, center=(rpatch_size // 2, rpatch_size // 2), 
    #           radius=rpatch_size//2, color=1, thickness=-1)
    #mask = np.expand_dims(mask, axis=-1)
    #
    # multiple dimension
    #mask_list = [ mask for i in range(channels)]
    #masks = np.concatenate(mask_list, axis=-1)     


    # params
    for fname, swath in swaths:
      # Generic: Old code when global mean & stdv does not apply
      # In the case, mean/stdv are necessary
      # Numpy nan option
      #if flag_nan:
      #  swath_mean = np.nanmean(swath, axis=(0,1))
      #  swath_std = np.nanstd(swath, axis=(0,1))
      #else :
      #  swath_mean = swath.mean(axis=(0,1))
      #  swath_std = swath.std(axis=(0,1))

      # Generic: Old code when global mean & stdv does not apply
      # modify small std value 
      #ill_stds = np.where(swath_std < 1.0e-20)[0]
      #if len(ill_stds) > 0 :
      #  for idx in ill_stds:
      #    swath_std[idx] += 1.0e-20

      print(fname)
      #print(swath.shape)
      max_x, max_y, _ = swath.shape

      # Shuffle patches
      coords = []
      for x in range(0, max_x, stride):
         for y in range(0, max_y, stride):
           if x + patch_size < max_x and y + patch_size < max_y:
              coords.append((x, y))
      # Baseically OFF
      #if flag_shuffle:
      #  np.random.shuffle(coords)
      
      # Get MOD35 data
      m35_file = get_filepath(fname, mod35_datadir, prefix='MOD35_L2.A')
      if not m35_file is None:
        hdf_m35 = SD(m35_file, SDC.READ)
        if hdf_m35 is None:
          # call mpiabort function
          raise NameError("MOD35 Path does not match any file")
          sys.excepthook =  mpiabort_excepthook
          sys.excepthook = sys.__excepthook__  # 2nd call?
      #if hdf_m35 is None:
      #  print(" Program Forcibly Terminate: Rank %d" % MPI.COMM_WORLD.Get_rank(), flush=True)
      #  MPI.COMM_WORLD.Abort()
        clouds_mask_img = gen_mod35_img(hdf_m35)

        for i, j in coords:
          patch = copy.deepcopy(swath[i:i + patch_size, j:j + patch_size])
          # Common standardization
          #if normalization:
          #  patch -= global_mean
          #  patch /= global_stdv
          # Assume band 6,7,20 are log-normal dist but 28,29 30 are normal-dist
          if normalization:
            if comm.Get_rank() == 0:
              print(" ## Apply log transformation for band 6,7,20## ") 

  
            for ichannel in range(channels):
              if ichannel >= 3:
                patch[:,:,:,ichannel] = patch[:,:,:,ichannel] - global_mean[ichannel]
                patch[:,:,:,ichannel] = patch[:,:,:,ichannel]/global_stdv[ichannel]
              else:
                patch[:,:,:,ichannel] = np.log10(patch[:,:,:,ichannel]+1.0e-10)
                
        
          if not np.isnan(patch).any():
          #TODO: Add lines below to compare MOD35
          # translate_const_clouds_array is based on const_clouds_array
          # Shape of clouds_patch [height, width, channels]
            clouds_patch, clouds_flag = translate_const_clouds_array(
              patch, clouds_mask_img, 
              width=patch_size, height=patch_size,thres=thres, coord=(i,j)
            )
            if clouds_flag:
              if not patch is None:
                if patch_size != rpatch_size:
                  # resize x mask
                  # apply resize
                  resize_clouds_patch = cv2.resize(
                    clouds_patch, dsize=(rpatch_size,rpatch_size), 
                    interpolation=cv2.INTER_AREA
                  )

                  # apply mask
                  return_clouds_patch = masks*resize_clouds_patch 
                  yield fname, (i, j), return_clouds_patch

                elif patch_size == rpatch_size:
                  # only apply mask
                  return_clouds_patch = masks*clouds_patch 
                  yield fname, (i, j), return_clouds_patch
                else:
                  print('Exception Error')
                  exit(0)
            

def write_feature(writer, filename, coord, patch):
    feature = {
        "filename": _bytes_feature(bytes(filename, encoding="utf-8")),
        "coordinate": _int64_feature(coord),
        "shape": _int64_feature(patch.shape),
        "patch": _bytes_feature(patch.ravel().tobytes()),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


    
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
            rec = "{}-{}_normed.tfrecord".format(rank, i // patches_per_record)
            #rec = "{}-{}.tfrecord".format(rank, i // patches_per_record)
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
    p.add_argument(
        "--shape",
        type=int,
        help="patch size. Assume Square image",
        default=128,
    )
    # NOT USED NOW
    p.add_argument(
        "--rshape",
        type=int,
        help="Resized shape 1024-->128",
    )
    p.add_argument(
        "--stride",
        type=int,
        help="patch stride. patch size/2 to compesate boundry information",
        default=64,
    )
    p.add_argument(
        "--ems_band",
        nargs='+',
        help="List of Emissive(Thermal/NearIR) additional band name. i.e. 28, 29 , 31  ",
    )
    p.add_argument(
        "--global_normalization", 
        type=int,
        help='If apply normalization by pre-computed mean&stdv for train data, 1. Otherwise(No-norm) 0',
        default=0
    )
    p.add_argument(
        "--stats_datadir", 
        type=str,
        help='If apply normalization, specify pre-computed stats info(mean&stdv) data directory',
        default='./'
    )
    p.add_argument(
        "--mod35_datadir", 
        type=str,
        help='MOD35_L2 data directory for training data',
        default='./'
    )
    p.add_argument(
        "--thres_cloud_frac", 
        type=float,
        help='threshold value range[0-1] for alignment process',
        default=0.3
    )
    
    # parse only one band info by int parser
    #p.add_argument(
    #    "--ems_band",
    #    type=str,
    #    help="Emissive(Thermal/NearIR) additional band name. i.e. 28 or 29 or 31  ",
    #    default='29',
    #)
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


def mpiabort_excepthook(type, value, traceback):
    mpi_comm = MPI.COMM_WORLD
    mpi_comm.Abort()
    sys.__excepthook__(type, value, traceback)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    FLAGS = get_args(verbose=rank == 0)
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # create filelist in each core
    fnames = []
    for i, f in enumerate(sorted(glob.glob(FLAGS.source_glob))):
        if i % size == rank:
            fnames.append(os.path.abspath(f))

    if not fnames:
        raise ValueError("source_glob does not match any files")

    # load global mean and stdv 
    if FLAGS.global_normalization == 1:
      for i in range(size):
        if i % size == rank:
           global_mean = np.load(glob.glob(FLAGS.stats_datadir+'/*_gmean.npy')[0])
           global_stdv = np.load(glob.glob(FLAGS.stats_datadir+'/*_gstdv.npy')[0])
           normalization = True
    else:
      normalization = False
      global_mean = np.zeros((6))
      global_stdv = np.ones((6))

    # process start
    #TODO make arg for ref_var & ems_var if using other modis dataset
    #TODO modify arg to add multiple temp/altitude bands  
    swaths  = gen_sds(fnames, 
                      ref_var='EV_500_Aggr1km_RefSB', ems_var='EV_1KM_Emissive',
                      ref_bands=["6","7"], ems_bands=FLAGS.ems_band)
                      #ref_bands=["6","7"], ems_bands=["20"]+FLAGS.ems_band)
    
    patches = gen_patches(swaths, 
                          stride=FLAGS.stride, 
                          patch_size=FLAGS.shape, 
                          rpatch_size=FLAGS.rshape,
                          channels=6,
                          normalization=normalization, 
                          global_mean=global_mean,
                          global_stdv=global_stdv,
                          mod35_datadir=FLAGS.mod35_datadir, 
                          thres=FLAGS.thres_cloud_frac )

    write_patches(patches, FLAGS.out_dir, FLAGS.patches_per_record)

    print("Rank %d done." % rank, flush=True)
