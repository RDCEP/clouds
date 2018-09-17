import sys
import os
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from tqdm import trange
from osgeo import gdal

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from reproduction.pipeline.load import add_pipeline_cli_arguments
#from reproduction.pipeline.load import preprocessed_pipeline
from reproduction.pipeline.load import load_data


def read_patches(files, height, width):
    batch = []
    for f in files:
        swath = gdal.Open(f)
        x_max = swath.RasterXSize - swath.RasterXSize % width
        y_max = swath.RasterYSize - swath.RasterYSize % height
        for x_off in range(0, x_max, width):
            for y_off in range(0, y_max, height):
                patch = swath.ReadAsArray(x_off, y_off, xsize=height, ysize=width)
                batch.append(np.rollaxis(patch, 0, 3))
                if len(batch) == 32:
                    yield np.stack(batch).astype(np.float32)
                    batch = []


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("mode", choices=["old_pipeline", "feed_dict", "new_pipeline"])
    p.add_argument("--data_glob")
    p.add_argument(
        "--epochs", type=int, help="Number of epochs to train for", default=5
    )
    p.add_argument(
        "--steps_per_epoch",
        metavar="N",
        help="Number of steps to train in each epoch ",
        type=int,
        default=1000,
    )

    FLAGS = p.parse_args()

    tif_files = [
        "data/tif/2017-01-01_MOD09GA_background_removal_zero_inputated_image_with_cf_50perc_grid_size10-0000017664-0000000000.tif",
        "data/tif/closed-open-cell-south-pacific.tif",
        "data/tif/open-cell-north-pacific.tif",
    ]
    records1 = [
        "data/tif/2017-01-01_MOD09GA_background_removal_zero_inputated_image_with_cf_50perc_grid_size10-0000017664-0000000000.tfrecord",
        "data/tif/closed-open-cell-south-pacific.tfrecord",
        "data/tif/open-cell-north-pacific.tfrecord",
    ]
    records2 = "data/tif2/*.tfrecord"
    # records2 = ["data/tif2/0-0.tfrecord"]

    if FLAGS.mode == "feed_dict":
        g = read_patches(tif_files, 64, 64)
        x = tf.placeholder(tf.float32, (32, 64, 64, 7))

    elif FLAGS.mode == "old_pipeline":

        _, dataset = _load_data(
            records1,
            fields=[f"b{i+1}" for i in range(7)],
            meta_json="data/tif/open-cell-north-pacific.json",
            shape=(64, 64),
            batch_size=32,
            normalization="whiten",
            read_threads=4,
            prefetch=1,
            shuffle_buffer_size=32,
        )
        f, c, x = dataset.make_one_shot_iterator().get_next()

    if FLAGS.mode == "new_pipeline":
        dataset = load_data(
            FLAGS.data_glob if FLAGS.data_glob else records2,
            shape=(64, 64, 7),
            batch_size=32,
            read_threads=4,
            shuffle_buffer_size=32,
            prefetch=1,
        )
        f, c, x = dataset.make_one_shot_iterator().get_next()

    y = tf.reduce_mean(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in trange(FLAGS.epochs):
            for step in trange(FLAGS.steps_per_epoch):
                feed_dict = {x: next(g)} if FLAGS.mode == "feed_dict" else None
                sess.run(y, feed_dict=feed_dict)
