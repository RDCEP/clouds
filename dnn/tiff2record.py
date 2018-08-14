from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mpi4py import MPI
from osgeo import gdal

import tensorflow as tf
import os
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def patch_reader(tiff_files, side_len):
    """Yields `side_len` by `side_len` sized chunks of every tiff file.
    Note there will be an edge effect to the right/bottom of each tiff.
    """
    for f in tiff_files:
        data = gdal.Open(f)
        rows = data.RasterXSize
        cols = data.RasterYSize

        rows -= rows % side_len
        cols -= cols % side_len

        for xoff in range(0, rows, side_len):
            for yoff in range(0, cols, side_len):
                data.ReadAsArray(xoff, yoff, side_len, side_len)
                data = np.rollaxis(data, 0, 3)

                if (img != 0).any():
                    yield img.astype(np.float32)


def whole_tif_reader(tiff_files):
    """Yields entire tif file read into numpy
    """
    for f in tiff_files:
        data = gdal.Open(f).ReadAsArray()
        data = np.rollaxis(data, 0, 3)
        yield data


def convert(data_gen, filename):
    """Converts a dataset to tfrecords.
    Args:
        data_gen: Iterator yielding 3D numpy arrays (chunks of tiff file)
        filename: Name of tf record output
    """
    count = 0
    with tf.python_io.TFRecordWriter(filename) as writer:
        for img in data_gen:
            rows, cols, bands = img.shape
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "rows": _int64_feature(rows),
                        "cols": _int64_feature(cols),
                        "bands": _int64_feature(bands),
                        "vals": _float_feature(img.flatten()),
                    }
                )
            )
            writer.write(example.SerializeToString())
            count += 1
    return count


def targets(tif_dir, size, rank):
    tifs = []
    for f in os.listdir(tif_dir):
        if f.split(".")[-1] == "tif":
            tifs.append(os.path.join(tif_dir, f))
    tifs.sort()
    return [f for i, f in enumerate(tifs) if i % size == rank]


if __name__ == "__main__":
    p = ArgumentParser(ArgumentDefaultsHelpFormatter)
    p.add_argument("tif_dir", help="Directory of tif files to extract from")
    p.add_argument("tfr_dir", help="Directory to put tf record")
    p.add_argument("--side_len", help="side length of each patch.", type=int)
    FLAGS = p.parse_args()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    tiff_files = targets(FLAGS.tif_dir, size, rank)
    tfr_file = os.path.join(FLAGS.tfr_dir, f"{rank + 1}_of_{size}.tfrecords")

    if FLAGS.side_len:
        tiff_imgs = patch_reader(tiff_files, FLAGS.side_len)
    else:
        tiff_imgs = whole_tif_reader(tiff_files)

    count = convert(tiff_imgs, tfr_file)
    print(f"rank {rank} finished and saved {count} patches.")
