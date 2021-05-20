#
#
#  Generate "SCALED" and "MASKED" tfrecord loading from RAW tfrecord data
#     i.e.  load un-normalized tfrecord, and then operate scaling [0,1] and circle mask
#
#
__author__ = "tkurihana@uchicago.edu"

import tensorflow as tf
import os
import gc
import sys
import cv2
import glob
import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import parallel_interleave
from tensorflow.contrib.data import batch_and_drop_remainder

from mpi4py   import MPI
from pathlib  import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# own library
homedir=str(Path.home())
#clouds_dir="/Research/clouds/src_analysis/lib_hdfs"
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


def data_extractor_fn(file,prefetch=1, read_threads=4, distribute=(1, 0)):
    def parser(ser):
        """
        Decode & Pass datast in tf.record
        *Cuation*
        floating point: tfrecord data ==> tf.float64
        """
        features = {
            "shape": tf.FixedLenFeature([3], tf.int64),
            "patch": tf.FixedLenFeature([], tf.string),
            "filename": tf.FixedLenFeature([], tf.string),
            "coordinate": tf.FixedLenFeature([2], tf.int64),
        }
        decoded = tf.parse_single_example(ser, features)
        patch = tf.reshape(
            tf.decode_raw(decoded["patch"], tf.float64), decoded["shape"]
        )
        # keep the value as float64

        # get other configs
        filename = decoded["filename"]
        coordinate = decoded["coordinate"]
        return patch, filename, coordinate
    
    dataset = (
        tf.data.Dataset.list_files([file], shuffle=True)
            .shard(*distribute)
            .apply(
            parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f).map(parser),
                cycle_length=read_threads,
                sloppy=True,
            )
        )
    )
    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()
    patches_list = []
    filename_list= []
    coord_list   = []
    crank = MPI.COMM_WORLD.Get_rank()
    with tf.Session() as sess:
        try:
            while True:
                patch, filename, coord  = sess.run(next_element)
                #i,j = coord
                #print(patch.shape, filename.decode('utf-8') , (i,j), flush=True)
                #yield patch, filename.decode('utf-8') , (i,j)
                patches_list.append(np.expand_dims(patch, axis=0))                
                filename_list.append(filename.decode('utf-8'))
                coord_list.append(coord)
        except tf.errors.OutOfRangeError:
          if crank == 0:
            print(" ###  TF-DEOCDED END--> next process ###", flush=True)
            pass

    patches = np.concatenate(patches_list, axis=0)
    coords = np.array(coord_list)
    return patches, filename_list, coords

def get_masks(rpatch_size, channels):

    mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float64)
    cv2.circle(mask, center=(rpatch_size // 2, rpatch_size // 2),
                radius=rpatch_size//2, color=1, thickness=-1)
    mask = np.expand_dims(mask, axis=-1)
    #  multiple dimension
    mask_list = [ mask for i in range(channels)]
    masks = np.concatenate(mask_list, axis=-1)
    return masks

def make_scale_mask_operator(patches, filename, coordinate, gmean, gstdv, channels=6,nsigma=2):
    """
      Apply [0,1] scaling and circle mask 

    """
    # scale [0,1]
    for ichannel in range(channels):
      ulim = gmean[ichannel]+nsigma*gstdv[ichannel]
      patches[:,:,:,ichannel] = patches[:,:,:,ichannel] / ulim

    # fill 1.0 where pixel values over nsigma
    upper_index = np.where(patches > 1.000)
    patches[upper_index] = 1.000

    # just in case remove negative values
    zero_index = np.where(patches < 0.000)
    if len(zero_index[0]) > 0:
      # Report when negative pixel was detected
      print(" #### !!!Negative Pixels Are Detected!!!", flush=True)
      patches[zero_index] = 0.00

    # circle mask
    _,patch_size,_,_  = patches.shape 
    mask = get_masks(patch_size,channels).reshape(1,patch_size,patch_size,channels)
    patches = patches * mask

    # return patches
    return filename, coordinate,patches

def make_scale_mask_sbands_operator(patches, filename, coordinate, gmean, gstdv, channels=6,nsigma=2, sbands=[0,1]):
    """
      Apply [0,1] scaling and circle mask 

    """
    # scale [0,1]
    for ichannel in range(channels):
      ulim = gmean[ichannel]+nsigma*gstdv[ichannel]
      patches[:,:,:,ichannel] = patches[:,:,:,ichannel] / ulim

    # fill 1.0 where pixel values over nsigma
    upper_index = np.where(patches > 1.000)
    patches[upper_index] = 1.000

    # just in case remove negative values
    zero_index = np.where(patches < 0.000)
    if len(zero_index[0]) > 0:
      # Report when negative pixel was detected
      print(" #### !!!Negative Pixels Are Detected!!!", flush=True)
      patches[zero_index] = 0.00

    # circle mask
    _,patch_size,_,_  = patches.shape 
    mask = get_masks(patch_size,channels).reshape(1,patch_size,patch_size,channels)
    patches = patches * mask

    # NEW select bandwidth
    if len( sbands ) > 1:
      spatches = np.concatenate( [ np.expand_dims(patches[:,:,:,int(iband)], axis=-1) for iband in sbands],axis=-1 )
    elif len(sbands) ==1:
      spatches = np.expand_dims(patches[:,:,:,int(sbands[0])], axis=-1)

    # return patches
    return filename, coordinate,spatches


def write_feature(writer, filename, coord, patch):
    feature = {
        "filename": _bytes_feature(bytes(filename, encoding="utf-8")),
        "coordinate": _int64_feature(coord),
        "shape": _int64_feature(patch.shape),
        "patch": _bytes_feature(patch.ravel().tobytes()),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


    
def write_patches(patches_info,basefname,out_dir, patches_per_record):
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
    filename, coords, patches = patches_info
    for i, patch in enumerate(zip(filename, coords, patches)):
        if i % patches_per_record == 0:
            rec = "{}".format(basefname)
            print("Writing to", rec, flush=True)
            f = tf.python_io.TFRecordWriter(os.path.join(out_dir, rec))

        write_feature(f, *patch)

        print("Rank", rank, "wrote", i+1, "patches", flush=True)

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
    p.add_argument(
        "--nsigma",
        type=int,
        help="upper bound of processing info",
        default=2,
    )
    p.add_argument(
        "--channels",
        type=int,
        help="number of channels in raw processed patch",
        default=6,
    )
    p.add_argument(
        "--stats_datadir", 
        type=str,
        help='If apply normalization, specify pre-computed stats info(mean&stdv) data directory',
        default='./'
    )
    p.add_argument(
        "--patches_per_record", type=int, help="Only used for pptif", default=500
    )
    # ONLY for sband selection
    p.add_argument(
        "--sbands",
        nargs='+',
        help="List of selected bandwidth from 5/6,7,20,28,29,31 ",
    )
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

    # load global mean and stdv 
    for i in range(size):
      if i % size == rank:
         gmean = np.load(glob.glob(FLAGS.stats_datadir+'/*_gmean.npy')[0])
         gstdv = np.load(glob.glob(FLAGS.stats_datadir+'/*_gstdv.npy')[0])

    # create filelist in each core
    fnames = []
    for i, f in enumerate(sorted(glob.glob(FLAGS.source_glob))):
      if i % size == rank:
        fname = os.path.abspath(f)
        #========================================================
        #                      process start
        #========================================================
        # Get patch info
        patch_info  = data_extractor_fn(fname)
        basefname = os.path.basename(fname)

        # Operate scaling and masking + yield patch info
        _patches, filenames, coordinates = patch_info
        # debug
        #print(np.mean(_patches, axis=(0,1,2)))

        ### Six bands
        patches = make_scale_mask_operator(
          _patches, filenames, coordinates,  gmean, gstdv, 
          channels=FLAGS.channels, nsigma=FLAGS.nsigma)

        ### Less bands
        #patches =  make_scale_mask_sbands_operator(
        #  _patches, filenames, coordinates,  gmean, gstdv, 
        #  channels=FLAGS.channels, nsigma=FLAGS.nsigma, sbands=FLAGS.sbands)
          #channels=FLAGS.channels, nsigma=FLAGS.nsigma)

        # Add 10/06
        npatches_per_record = _patches.shape[0]
        print(f" tfname {os.path.basename(fname)} | num. of patches {npatches_per_record} ", flush=True)
    
        # save into tfrecord
        #write_patches(patches, basefname, FLAGS.out_dir, FLAGS.patches_per_record)
        write_patches(patches, basefname, FLAGS.out_dir, npatches_per_record)
        gc.collect()

        print("Rank %d done." % rank, flush=True)
