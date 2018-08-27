"""
Routine to convert MOD06_L2 HDF swath files to TFRecord format
"""

import tensorflow as tf
from pyhdf.SD import SD, SDC
from statistics import median
import json
import numpy as np
from mpi4py import MPI
from os import path
from argparse import ArgumentParser
import os

# from IPython import embed  # DEBUG


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def hdf2tfr(hdf_file, target_fields, as_bytes, with_meta):
    """Converts HDF file into a tf record by serializing all fields and
    names to a record. Also outputs a json file holding the meta data.
    Arguments:
        hdf_file: File to convert into a tf record
        target_fields: List of specified fields to convert
        as_bytes: Write data as bytes (as opposed to converting everything to floats)
        with_meta: Keep shape and type information in tfrecord too
    """
    data = SD(hdf_file, SDC.READ)
    meta = {}
    tfr_features = {}

    for field in data.datasets().keys():

        if target_fields and field not in target_fields:
            continue

        values = data.select(field)[:]  # Index to get it as np array

        # Make sure every field is of rank 4
        while len(values.shape) < 3:
            values = np.expand_dims(values, -1)
        if len(values.shape) > 3:
            print(
                "Warning, encountered high rank field %s with shape%s"
                % (field, values.shape)
            )
            continue

        # Reorder dimensions such that it is longest dimension first
        rank_order = list(np.argsort(values.shape))
        rank_order.reverse()
        values = values.transpose(rank_order)
        meta[field] = [*values.shape, str(values.dtype)]

        if with_meta:
            height, width, channels, = values.shape
            tfr_features[field + "/shape"] = _int64_feature(values.shape)
            # FIXME field unused if not `as_bytes` since type is cast to float
            ty = str(values.dtype).encode("utf_8")
            tfr_features[field + "/type"] = _bytes_feature(ty)

        if as_bytes:
            x = values.ravel().tobytes()
            tfr_features[field] = _bytes_feature(x)
        else:
            x = values.astype(np.float32).ravel()
            tfr_features[field] = _float_feature(x)

    example = tf.train.Example(features=tf.train.Features(feature=tfr_features))

    base, _ = path.splitext(hdf_file)
    json_file = base + ".json"
    tfr_file = base + ".tfrecord"

    if not with_meta:
        with open(json_file, "w") as f:
            json.dump(meta, f)

    with tf.python_io.TFRecordWriter(tfr_file) as f:
        f.write(example.SerializeToString())


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--hdf_dir", required=True)
    p.add_argument(
        "--fields",
        nargs="+",
        help="Specific fields to record. If none are provided then all fields are recorded.",
    )
    p.add_argument(
        "--bytes",
        action="store_true",
        help="each field is stored as bytes as opposed to cast to float32",
    )
    p.add_argument(
        "--with_meta",
        action="store_true",
        help="Store field metadata inside of tfrecord",
    )

    FLAGS = p.parse_args()
    FLAGS.hdf_dir = path.abspath(FLAGS.hdf_dir)

    for f in FLAGS.__dict__:
        print(f"\t{f}:{(25-len(f)) * ' '} {FLAGS.__dict__[f]}")
    print("\n")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    targets = os.listdir(FLAGS.hdf_dir)
    targets = [path.join(FLAGS.hdf_dir, t) for t in targets if t[-4:] == ".hdf"]
    targets.sort()
    targets = [t for i, t in enumerate(targets) if i % size == rank]
    for t in targets:
        # HDFtoTFRecord(FLAGS.out_dir, t, '.hdf')
        hdf2tfr(path.join(FLAGS.hdf_dir, t), FLAGS.fields, FLAGS.bytes, FLAGS.with_meta)
