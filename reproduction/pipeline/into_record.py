import tensorflow as tf
import os
import cv2
import json
import glob
import numpy as np

from osgeo import gdal
from mpi4py import MPI
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pyhdf.SD import SD, SDC


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def gen_swaths(targets, mode, resize, rank):
    """Reads and yields resized swaths.
    """
    if mode == "tif":
        read = lambda tif_file: gdal.Open(tif_file).ReadAsArray()

    elif mode == "mod02_1km":
        read = mod02_1km_read

    else:
        raise ValueError("Invalid reader mode", mode)

    for t in targets:
        print("rank", rank, "reading", t, flush=True)
        swath = np.rollaxis(read(t), 0, 3)
        if resize is not None:
            swath = cv2.resize(
                swath, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA
            )
        yield t, swath


def mod02_1km_read(hdf_file):
    """Read `hdf_file` and extract relevant fields as per `names_1km`.
    """
    hdf = SD(hdf_file, SDC.READ)
    names_1km = (
        "EV_250_Aggr1km_RefSB",
        "EV_500_Aggr1km_RefSB",
        "EV_1KM_RefSB",
        "EV_1KM_Emissive",
    )
    fields = [hdf.select(n)[:] for n in names_1km]
    return np.concatenate(fields, axis=0)


def gen_patches(swaths, shape, strides):
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

    print("Rank", rank, "wrote", i + 1, "patches", flush=True)


def get_args(verbose=False):
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Turns tif or hdf data into tfrecords. TODO better description.",
    )
    p.add_argument("source_glob", help="Glob of files to convert to tfrecord")
    p.add_argument("out_dir", help="Directory to save results")
    p.add_argument(
        "mode",
        choices=["mod09_tif", "mod02_1km"],
        help=(
            "`mod09_tif`: Turn whole .tif swath into tfrecord. "
            "`mod02_1km` : Extracts EV_250_Aggr1km_RefSB, EV_500_Aggr1km_RefSB, "
            "EV_1KM_RefSB, and EV_1KM_Emissive."
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
        "--patches_per_record", type=int, help="Only used for pptif", default=500
    )

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

    targets = []
    for i, f in enumerate(sorted(glob.glob(FLAGS.source_glob))):
        if i % size == rank:
            targets.append(os.path.abspath(f))

    swaths = gen_swaths(targets, FLAGS.mode, FLAGS.resize, rank)
    patches = gen_patches(swaths, FLAGS.shape, FLAGS.stride)
    write_patches(rank, patches, FLAGS.out_dir, FLAGS.patches_per_record)

    print("Rank %d done." % rank, flush=True)
