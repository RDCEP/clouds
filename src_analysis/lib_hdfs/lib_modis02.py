import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.contrib.data import shuffle_and_repeat
from tensorflow.contrib.data import parallel_interleave
from tensorflow.contrib.data import batch_and_drop_remainder

def load_tfrecord(serialized_data,
                shape,
                batch_size=1,
                read_threads=4,
                shuffle_buffer_size=1000,
                prefetch=1,
                distribute=(1, 0)
               ):
    
    def parser(serialized_data):
        features = {
            "shape": tf.FixedLenFeature([3], tf.int64),
            "patch": tf.FixedLenFeature([], tf.string),
            "filename":tf.FixedLenFeature([],tf.string),
            "coordinate":tf.FixedLenFeature([2],tf.int64),
        }
        decoded = tf.parse_single_example(serialized_data,features)
        # &&&&&& My output id tf.float64 !!!! &&&&&&
        patch   = tf.reshape(tf.decode_raw(decoded["patch"], tf.float64), decoded["shape"])
        # randomly crop mini-patches from data
        patch = tf.random_crop(patch, shape)
        print(decoded["filename"], decoded["coordinate"], patch)
        return decoded["filename"], decoded["coordinate"], patch
 
    # TODO: understand this code
    dataset = (
    tf.data.Dataset.list_files(serialized_data, shuffle=True)
    .shard(*distribute)
    .apply(
        parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f).map(parser),
                cycle_length=read_threads,
                sloppy=True,
            )
        )
    )

    # TODO: understand the code
    print(dataset)
    dataset = dataset.apply(batch_and_drop_remainder(batch_size)).prefetch(prefetch)
    return dataset

def _get_imgs(dataset,ae=None,fields=None, n=500):
    # get data from dataset
    names, coords, imgs = [], [], []
    batch = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        while len(imgs) < n:
            names_, coords_, imgs_ = sess.run(batch)
            names.extend(names_)
            coords.extend(coords_)
            imgs.extend(imgs_)
        return np.asarray(imgs), coords, names

def _get_num_imgs(tfrecord):
    count=0
    for irecord in tf.python_io.tf_record_iterator(tfrecord):
        count += 1
    return count

def proc_sds(sds_array):
    """
    IN: array = hdf_data.select(variable_name)
    """
    array = sds_array.get()
    array = array.astype(np.float64)
    
    # check bandinfo
    _bands = sds_array.attributes()['band_names']
    print("Process bands", _bands)
    bands = _bands.split(",")
    
    # nan process
    nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
    if len(nan_idx) > 0:
        array[nan_idx] = np.nan
    else:
        pass
    # invalid value process
    # TODO: future change 32767 to other value
    invalid_idx = np.where( array > 32767 )
    if len(nan_idx) > 0:
        array[invalid_idx] = np.nan
    else:
        pass
    
    # radiacne offset
    offset = sds_array.attributes()['radiance_offsets']
    offset_array = np.zeros(array.shape) # new matrix
    offset_ones  = np.ones(array.shape)  # 1 Matrix 
    offset_array[:,:] = array[:,:] - offset*offset_ones[:,:]
    
    # radiance scale
    scales = sds_array.attributes()['radiance_scales']
    scales_array = np.zeros(array.shape) # new matrix
    scales_array[:,:] = scales*offset_array[:,:]
    return scales_array, bands

