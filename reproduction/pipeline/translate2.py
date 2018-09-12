import tensorflow as tf
import numpy as np
from osgeo import gdal
from argparse import ArgumentParser
from into_record import _int64_feature, _bytes_feature, _float_feature
from mpi4py import MPI
import os


def read_tif(files, shape, strides):
    height, width = shape
    stride_x, stride_y = strides

    for f in files:
        print("Reading from", f, flush=True)
        swath = gdal.Open(f).ReadAsArray()
        swath = np.rollaxis(swath, 0, 3).astype(np.float32)
        normalized = (swath - swath.mean(axis=(0, 1))) / swath.std(axis=(0, 1))
        max_x, max_y, _ = swath.shape

        coordinates = []
        for x in range(0, max_x, stride_x):
            for y in range(0, max_y, stride_y):
                if x + width < max_x and y + height < max_y:
                    coordinates.append((x, y))
        np.random.shuffle(coordinates)

        for x, y in coordinates:
            patch = swath[x : x + width, y : y + height].copy()
            if (patch[0, 0, 0] != patch).any() and not np.isnan(patch).any():
                yield f, (x, y), patch


def write(tfr_dir, patches, rank, patches_per_record=1000):

    for i, (filename, coord, patch) in enumerate(patches):
        if i % patches_per_record == 0:
            rec = "{}-{}.tfrecord".format(rank, i // patches_per_record)
            print("Writing to", rec, flush=True)
            f = tf.python_io.TFRecordWriter(os.path.join(tfr_dir, rec))

        feature = {
            "filename": _bytes_feature(bytes(filename, encoding="utf-8")),
            "coordinate": _int64_feature(coord),
            "shape": _int64_feature(patch.shape),
            "patch": _float_feature(patch.ravel()),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        f.write(example.SerializeToString())


def get_args():
    p = ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("save_dir")
    p.add_argument("--shape", nargs=2, type=int, default=(256, 256))
    p.add_argument("--strides", nargs=2, type=int, default=(128, 128))
    return p.parse_args()


def main():
    FLAGS = get_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    targets = []
    for t in os.listdir(FLAGS.data_dir):
        if t[-4:] == ".tif":
            targets.append(os.path.join(FLAGS.data_dir, t))
    targets.sort()
    targets = [t for i, t in enumerate(targets) if i % size == rank]

    patches = read_tif(targets, FLAGS.shape, FLAGS.strides)
    write(FLAGS.save_dir, patches, rank)


if __name__ == "__main__":
    main()
