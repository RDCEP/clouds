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
from osgeo import gdal

# from IPython import embed  # DEBUG


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def save_example_and_metadata(original, features, meta):
    """Saves tf record from examples and json meta data using original name but
    changing file extension.
    """
    examples = [
        tf.train.Example(features=tf.train.Features(feature=f)) for f in features
    ]

    base, _ = path.splitext(original)
    json_file = base + ".json"
    tfr_file = base + ".tfrecord"

    with open(json_file, "w") as f:
        json.dump(meta, f)

    with tf.python_io.TFRecordWriter(tfr_file) as f:
        for example in examples:
            f.write(example.SerializeToString())


def tif2tfr(tif_file, out_dir, stride=2048):
    tif = gdal.Open(tif_file)

    x_tot = tif.RasterXSize
    y_tot = tif.RasterYSize
    feature_list = []
    meta = {}

    for x_off in range(0, x_tot, stride):
        for y_off in range(0, y_tot, stride):
            features = {}
            sx = min(stride, x_tot - x_off)
            sy = min(stride, y_tot - y_off)
            data = tif.ReadAsArray(x_off, y_off, sx, sy)
            n_bands, height, width = data.shape

            for b in range(n_bands):
                field = "b%d" % (b + 1)
                meta[field] = [1, str(data.dtype)]
                features[field + "/shape"] = _int64_feature([height, width, 1])
                features[field] = _bytes_feature(data[b].tobytes())

            feature_list.append(features)

    out_file = path.join(out_dir, path.basename(tif_file))
    save_example_and_metadata(out_file, feature_list, meta)


def hdf2tfr(hdf_file, out_dir, target_fields):
    """Converts HDF file into a tf record by serializing all fields and
    names to a record. Also outputs a json file holding the meta data.
    Arguments:
        hdf_file: File to convert into a tf record
        target_fields: List of specified fields to convert
    """
    hdf = SD(hdf_file, SDC.READ)
    meta = {}
    features = {}

    for field in hdf.datasets().keys():
        if target_fields and field not in target_fields:
            continue

        data = hdf.select(field)[:]  # Index to get it as np array

        # Make sure every field is of rank 4
        while len(data.shape) < 3:
            data = np.expand_dims(data, -1)
        if len(data.shape) > 3:
            print(
                "Warning, encountered high rank field %s with shape %s"
                % (field, data.shape)
            )
            continue

        # Reorder dimensions such that it is longest dimension first
        # Assumes height > width > channels
        rank_order = list(np.argsort(data.shape))
        rank_order.reverse()
        data = data.transpose(rank_order)

        ty = str(data.dtype)
        meta[field] = [data.shape[-1], ty]
        features[field + "/shape"] = _int64_feature(data.shape)
        features[field + "/type"] = _bytes_feature(ty.encode("utf_8"))
        features[field] = _bytes_feature(data.tobytes())

    out_file = path.join(out_dir, path.basename(hdf_file))
    save_example_and_metadata(out_file, [features], meta)


def get_args():
    p = ArgumentParser(
        description=(
            "Reads HDF or TIF files and translates them into tf records and json "
            "meta data files. Parallelized with MPI4py.\n"
            "The tfrecord format is "
            "{'field': bytes, 'field/shape': [height, width, channels]}, where "
            "height, width, and channels are int64. The JSON format is "
            "{'field': [channels, type]}."
        )
    )
    p.add_argument("--hdf_dir")
    p.add_argument("--tif_dir")
    p.add_argument(
        "--out_dir",
        help=(
            "Directory to save results, if unprovided, then tfrecords and json "
            "meta data are saved to the same directory as the source data."
        ),
    )
    p.add_argument(
        "--fields",
        nargs="+",
        help=(
            "This is only used when translating hdf files, it specifies which "
            "fields to record. If none are provided then all fields are "
            "recorded. For tif files, all fields are translated as b0..bN and "
            "this flag is ignored."
        ),
    )

    FLAGS = p.parse_args()

    if bool(FLAGS.hdf_dir) == bool(FLAGS.tif_dir):
        print("Please specify either --hdf_dir or --tif_dir.")
        exit(1)

    is_hdf = bool(FLAGS.hdf_dir)
    data_dir = path.abspath(FLAGS.hdf_dir if is_hdf else FLAGS.tif_dir)

    for f in FLAGS.__dict__:
        print(f"\t{f}:{(25-len(f)) * ' '} {FLAGS.__dict__[f]}")
    print("\n")

    out_dir = path.abspath(FLAGS.out_dir) if FLAGS.out_dir else data_dir

    return data_dir, is_hdf, FLAGS.fields, out_dir


if __name__ == "__main__":
    data_dir, is_hdf, fields, out_dir = get_args()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    ext = ".hdf" if is_hdf else ".tif"
    targets = [t for t in os.listdir(data_dir) if t[-4:] == ext]
    targets.sort()
    targets = [t for i, t in enumerate(targets) if i % size == rank]
    os.makedirs(out_dir, exist_ok=True)

    for t in targets:
        print("Rank %d converting %s" % (rank, t), flush=True)
        try:
            if is_hdf:
                hdf2tfr(path.join(data_dir, t), out_dir, fields)
            else:
                tif2tfr(path.join(data_dir, t), out_dir)
        except:
            print("Rank %d Failed to serialize %s" % (rank, t), flush=True)

    print("Rank %d done." % rank, flush=True)
