import tensorflow as tf
import os
import cv2
import json
import numpy as np

from os import path
from osgeo import gdal
from mpi4py import MPI
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


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


def normalized_patches(tif_files, shape, strides, resize):
    """Generates normalized patches.
    """
    for tif_file in tif_files:
        print("Reading", tif_file, flush=True)
        swath = gdal.Open(tif_file).ReadAsArray()
        swath = np.rollaxis(swath, 0, 3)
        # TODO Flags for other kinds of normalization
        # NOTE: Normalizing the whole (sometimes 8gb) swath will double memory usage by
        # casting it from int16 to float32. Instead normalize and cast patches.
        if resize is not None:
            swath = cv2.resize(
                swath, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA
            )
        mean = swath.mean(axis=(0, 1)).astype(np.float32)
        std = swath.std(axis=(0, 1)).astype(np.float32)
        max_x, max_y, _ = swath.shape
        stride_x, stride_y = strides
        shape_x, shape_y = shape

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
            if has_clouds and not np.isnan(patch).any():
                patch = (patch.astype(np.float32) - mean) / std
                yield tif_file, (x, y), patch


def write_patches(rank, patches, out_dir, patches_per_record):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
    """
    i = 0
    for i, (filename, coord, patch) in enumerate(patches):
        if i % patches_per_record == 0:
            rec = "{}-{}.tfrecord".format(rank, i // patches_per_record)
            print("Writing to", rec, flush=True)
            f = tf.python_io.TFRecordWriter(os.path.join(out_dir, rec))
        feature = {
            "filename": _bytes_feature(bytes(filename, encoding="utf-8")),
            "coordinate": _int64_feature(coord),
            "shape": _int64_feature(patch.shape),
            "patch": _bytes_feature(patch.ravel().tobytes()),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        f.write(example.SerializeToString())

    print("Rank", rank, "wrote", i + 1, "patches")


def get_args():
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Turns tif or hdf data into tfrecords. TODO better description.",
    )
    p.add_argument("source_dir", help="Directory of files to convert to tfrecord")
    p.add_argument(
        "mode",
        choices=["tif", "hdf", "pptif"],
        help=(
            "`tif`: Turn whole .tif swath into tfrecord. "
            "`hdf`: Turn .hdf swath into tfrecord (respecting fields). "
            "`pptif`: preprocessed_tif, normalize and patchify a tif file."
        ),
    )
    p.add_argument(
        "--out_dir",
        help=(
            "Directory to save results, if unprovided, then tfrecords and json meta data "
            "are saved to the same directory as the source data."
        ),
    )
    p.add_argument(
        "--fields",
        nargs="+",
        help=(
            "This is only used when translating hdf files, it specifies which fields to "
            "record. If none are provided then all fields are recorded. For tif files, "
            "all fields are translated as b0..bN and this flag is ignored."
        ),
    )
    p.add_argument(
        "--shape",
        nargs=2,
        type=int,
        help="patch shape. Only used for pptif",
        default=(128, 128),
    )
    p.add_argument(
        "--resize",
        type=float,
        help="Resize fraction e.g. 0.25 to quarter scale. Only used for pptif",
    )
    p.add_argument(
        "--stride",
        nargs=2,
        type=int,
        help="patch stride. Only used for pptif",
        default=(64, 64),
    )
    p.add_argument(
        "--patches_per_record", type=int, help="Only used for pptif", default=5000
    )

    FLAGS = p.parse_args()

    for f in FLAGS.__dict__:
        print("\t", f, (25-len(f)) * " ", FLAGS.__dict__[f])
    print("\n")

    FLAGS.data_dir = path.abspath(FLAGS.source_dir)
    FLAGS.out_dir = path.abspath(FLAGS.out_dir) if FLAGS.out_dir else FLAGS.data_dir
    return FLAGS


def get_targets(data_dir, ext, rank):
    targets = [t for t in os.listdir(data_dir) if t[-4:] == ext]
    targets.sort()
    return [path.join(data_dir, t) for i, t in enumerate(targets) if i % size == rank]


if __name__ == "__main__":
    FLAGS = get_args()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    if FLAGS.mode == "hdf":
        for t in get_targets(FLAGS.data_dir, ".hdf", rank):
            print("Rank", rank, "Converting", t)
            hdf2tfr(t, FLAGS.out_dir, FLAGS.fields)

    elif FLAGS.mode == "tif":
        for t in get_targets(FLAGS.data_dir, ".tif", rank):
            print("Rank", rank, "Converting", t)
            tif2tfr(t, FLAGS.out_dir)

    elif FLAGS.mode == "pptif":
        targets = get_targets(FLAGS.data_dir, ".tif", rank)
        patches = normalized_patches(targets, FLAGS.shape, FLAGS.stride, FLAGS.resize)
        write_patches(rank, patches, FLAGS.out_dir, FLAGS.patches_per_record)

    else:
        raise ValueError("Invalid mode")

    print("Rank %d done." % rank, flush=True)
