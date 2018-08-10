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


def reader(tiff_files, side_len):
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
                bands = []
                for b in range(data.RasterCount):
                    band = data.GetRasterBand(b + 1).ReadAsArray(
                        xoff=xoff, yoff=yoff, win_xsize=side_len, win_ysize=side_len
                    )
                    bands.append(band)

                if any(band is None for band in bands):
                    raise IndexError(
                        f"In {f} at {(xoff, yoff)} with side_len {side_len}"
                    )

                img = np.stack(bands, axis=-1)
                if (img != 0).any():
                    yield img.astype(np.float32)


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
    tifs = os.listdir(tif_dir)
    tifs.sort()
    return [os.path.join(tif_dir, f) for i, f in enumerate(tifs) if i % size == rank]


if __name__ == "__main__":
    p = ArgumentParser(ArgumentDefaultsHelpFormatter)
    p.add_argument("tif_dir", help="Directory of tif files to extract from")
    p.add_argument("tfr_dir", help="Directory to put tf record")
    p.add_argument(
        "--side_len", help="side length of each patch.", type=int, default=64
    )
    FLAGS = p.parse_args()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    tiff_files = targets(FLAGS.tif_dir, size, rank)

    tiff_imgs = reader(tiff_files, FLAGS.side_len)

    tfr_file = os.path.join(FLAGS.tfr_dir, f"{rank + 1}_of_{size}.tfrecords")

    count = convert(tiff_imgs, tfr_file)
    print(f"rank {rank} finished and saved {count} patches.")
