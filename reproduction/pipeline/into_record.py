import tensorflow as tf

# from pyhdf.SD import SD, SDC
from statistics import median
import json
import numpy as np
from mpi4py import MPI
from os import path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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


def normalized_patches(tif_files, shape, strides):
    """Generates normalized patches.
    """
    for tif_file in tif_files:
        print("Reading", tif_file, flush=True)
        swath = gdal.Open(tif_file).ReadAsArray()
        swath = np.rollaxis(swath, 0, 3)
        # TODO Flags for other kinds of normalization
        # NOTE: Normalizing the whole (sometimes 8gb) swath will double memory usage by
        # casting it from int16 to float32. Instead normalize and cast patches.
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
            patch = swath[x : x + shape_x, y : y + shape_y].astype(np.float32)
            # Filter away patches with Nans or no clouds (ie whole patch 1 value)
            if (patch[0, 0, 0] != patch).any() and not np.isnan(patch).any():
                patch = (patch - mean) / std
                yield tif_file, (x, y), patch


def write_patches(rank, patches, out_dir, patches_per_record):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
    """
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
        print(f"\t{f}:{(25-len(f)) * ' '} {FLAGS.__dict__[f]}")
    print("\n")

    FLAGS.data_dir = path.abspath(FLAGS.source_dir)
    FLAGS.out_dir = path.abspath(FLAGS.out_dir) if FLAGS.out_dir else data_dir
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
        patches = normalized_patches(targets, FLAGS.shape, FLAGS.stride)
        write_patches(rank, patches, FLAGS.out_dir, FLAGS.patches_per_record)

    else:
        raise ValueError("Invalid mode")

    print("Rank %d done." % rank, flush=True)
