"""Iterate through tfrecords, encode the examples and save the encodings.

Encodings are saved in the format expected by parallel-kmeans. First 8 bytes of the file
are two int32 specifying number of vectors and their dimensionality. The rest of the file
are float32s representing the values of those vectors in C order.
"""
__author__ = "casperneo@uchicago.edu"

from argparse import ArgumentParser

import os, sys
import tensorflow as tf
import numpy as np
from utils import load_model_def, load_latest_model_weights
from pipeline import load


p = ArgumentParser(description=__doc__)
p.add_argument("out_file")
p.add_argument("encoder")
p.add_argument("--latent", choices=["flatten", "spatial_mean"], default="spatial_mean")
load.add_pipeline_cli_arguments(p)
FLAGS = p.parse_args()

for f in FLAGS.__dict__:
    print(f, (25 - len(f)) * " ", FLAGS.__dict__[f])
print("\n", flush=True)


ds = load.load_data(
    FLAGS.data,
    FLAGS.shape,
    FLAGS.batch_size,
    FLAGS.read_threads,
    FLAGS.shuffle_buffer_size,
    FLAGS.prefetch,
    not FLAGS.no_augment_flip,
    not FLAGS.no_augment_rotate,
    repeat=False,
)

encoder = load_model_def(FLAGS.encoder, "encoder")

_, _, imgs = ds.make_one_shot_iterator().get_next()
codes = encoder(imgs)

if FLAGS.latent == "spatial_mean":
    codes = tf.reduce_mean(codes, axis=(1, 2))

elif FLAGS.latent == "flatten":
    codes = tf.reshape(codes, [FLAGS.batch_size, -1])

else:
    raise ValueError("Invalid latent vector treatment", FLAGS.latent)

print("Starting session", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    load_latest_model_weights(encoder, FLAGS.encoder, "encoder")

    with open(FLAGS.out_file, "wb") as f:
        print("Writing", flush=True)
        # first int32 is number of lines, second i32 is dimensionality
        f.write(np.array([0, 0], dtype=np.int32).tobytes())

        try:
            count, dims = 0, None
            while True:
                c = sess.run(codes)
                b, d = c.shape

                if dims is not None:
                    assert dims == d, "Dimensions inconsistent."
                else:
                    dims = d

                f.write(c.astype(np.float32).ravel().tobytes())
                count += b

        except tf.errors.OutOfRangeError:
            f.seek(0)
            f.write(np.array([count, dims], dtype=np.int32).tobytes())
            print("Finished. count", count, "dims", dims)
