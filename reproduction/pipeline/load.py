"""This library contains the functions and flags related to loading clouds datasets.
"""
__author__ = "casperneo@uchicago.edu"


import tensorflow as tf
import numpy as np
import json
from tensorflow.contrib.data import shuffle_and_repeat
from tensorflow.contrib.data import parallel_interleave
from tensorflow.contrib.data import batch_and_drop_remainder


def load_data_direct(
    data_glob,
    shape,
    fields,
    normalization_file,
    batch_size=32,
    flip=True,
    rotate=True,
    shuffle_buffer_size=1000,
    distribute=(1, 0),
    read_threads=4,
    prefetch=1,
    repeat=True,
):
    # TODO direct load of tif
    parser = open_and_normalize_hdf(shape, fields, normalization_file, flip, rotate)
    data = (
        tf.data.Dataset.list_files(data_glob, shuffle=True).shard(*distribute)
        # OPTIMIZE: Does read-threads do anything? Considering Python GLC? TODO test.
        .map(parser, num_parallel_calls=read_threads)
    )
    if repeat:
        data = data.apply(shuffle_and_repeat(shuffle_buffer_size))
    else:
        data = data.shuffle(shuffle_buffer_size)

    data = data.apply(batch_and_drop_remainder(batch_size)).prefetch(prefetch)
    return data


def open_and_normalize_hdf(shape, fields, normalization_file, flip, rotate):
    """Returns a function that opens and normalizes a random patch from fname, returning
    fname, central coordinate, and patch.
    """
    with open(normalization_file, "r") as f:
        normalization = json.load(f)

    def fn(fname):
        width, height, _ = shape
        if rotate:
            width = int(np.ceil(width * 2 ** 0.5))
            height = int(np.ceil(height * 2 ** 0.5))

        def pyfn(fname):
            hdf = SD(str(fname)[2:-1], SDC.READ)
            b = hdf.select(fields[0])
            h = np.random.randint(
                0, b.dimensions()["10*nscans:MODIS_SWATH_Type_L1B"] - height
            )
            w = np.random.randint(
                0, b.dimensions()["Max_EV_frames:MODIS_SWATH_Type_L1B"] - width
            )
            coord = np.array([h + height // 2, w + width // 2], dtype=np.int64)

            patch = []

            for f in fields:
                x = hdf.select(f)[:, h : h + height, w : w + width]
                patch.append(x)
            patch = np.concatenate(patch)
            patch = np.rollaxis(patch, 0, 3).astype(np.float32)
            patch -= np.array(normalization["sub"], dtype=np.float32)
            patch /= np.array(normalization["div"], dtype=np.float32)

            return fname, coord, patch

        fname, coord, patch = tf.py_func(
            pyfn, [fname], (tf.string, tf.int64, tf.float32)
        )
        coord.set_shape(2)
        patch.set_shape([height, width, shape[2]])

        if rotate:
            angle = tf.random_uniform((), 0, 6.28)
            patch = tf.contrib.image.rotate(patch, angle)
            patch = tf.image.central_crop(patch, 2 ** -0.5)

        if flip:
            patch = tf.image.random_flip_up_down(tf.image.random_flip_left_right(patch))

        return fname, coord, patch

    return fn


def load_data(
    data_glob,
    shape,
    batch_size=32,
    read_threads=4,
    shuffle_buffer_size=1000,
    prefetch=1,
    flips=True,
    rotate=False,
    distribute=(1, 0),
    repeat=True,
):
    """Returns a dataset of (filenames, coordinates, patches).
    See `add_pipeline_cli_arguments` for argument descriptions.
    """

    #TODO add parser for floating point
    def parser(ser):
        features = {
            "shape": tf.FixedLenFeature([3], tf.int64),
            "patch": tf.FixedLenFeature([], tf.string),
            "filename": tf.FixedLenFeature([], tf.string),
            "coordinate": tf.FixedLenFeature([2], tf.int64),
        }
        decoded = tf.parse_single_example(ser, features)
        patch = tf.reshape(
            tf.decode_raw(decoded["patch"], tf.float64), decoded["shape"]
            #tf.decode_raw(decoded["patch"], tf.float32), decoded["shape"]
        )
        if rotate:
            angle = tf.random_uniform((), 0, 6.28)
            patch = tf.contrib.image.rotate(patch, angle)
            patch = tf.image.central_crop(patch, 2 ** -0.5)

        patch = tf.random_crop(patch, shape)
        if flips:
            patch = tf.image.random_flip_up_down(tf.image.random_flip_left_right(patch))
        return decoded["filename"], decoded["coordinate"], patch

    dataset = (
        tf.data.Dataset.list_files(data_glob, shuffle=True)
        .shard(*distribute)
        .apply(
            parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f).map(parser),
                cycle_length=read_threads,
                sloppy=True,
            )
        )
    )
    if repeat:
        dataset = dataset.apply(shuffle_and_repeat(shuffle_buffer_size))
    else:
        dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.apply(batch_and_drop_remainder(batch_size)).prefetch(prefetch)

    return dataset


def add_pipeline_cli_arguments(p):
    """Adds flags used for loading an argparse.ArgumentParser.
    """
    p.add_argument(
        "--data", help="patterns to pick up tf records files", required=True, nargs="+"
    )
    p.add_argument(
        "--shape",
        nargs=3,
        type=int,
        metavar=("h", "w", "c"),
        help="Shape of input image",
        default=(64, 64, 7),
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--read_threads", type=int, default=4, metavar="num_threads")
    p.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Size of prefetch buffers in dataset pipeline",
    )
    p.add_argument("--shuffle_buffer_size", type=int, default=1000)
    p.add_argument("--no_augment_flip", action="store_true", default=False)
    p.add_argument("--no_augment_rotate", action="store_true", default=False)
