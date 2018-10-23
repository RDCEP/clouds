from argparse import ArgumentParser

import os, sys
import tensorflow as tf
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from reproduction.pipeline import load


p = ArgumentParser()
p.add_argument("out_file")
p.add_argument("encoder")
p.add_argument("--latent", choices=["flatten", "spatial_mean"], default="spatial_mean")
p.add_argument("--max_steps", type=int, default=100000)
p.add_argument("--n_clusters", type=int, default=60)
p.add_argument("--summary_every", type=int, default=10)
p.add_argument("--save_every", type=int, default=50)
load.add_pipeline_cli_arguments(p)
FLAGS = p.parse_args()


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

with open(os.path.join(FLAGS.encoder, "encoder.json"), "r") as f:
    encoder = tf.keras.models.model_from_json(f.read())

_, _, imgs = ds.make_one_shot_iterator().get_next()
codes = encoder(imgs)

if FLAGS.latent == "spatial_mean":
    codes = tf.reduce_mean(codes, axis=(1, 2))

elif FLAGS.latent == "flatten":
    codes = tf.reshape(codes, [FLAGS.batch_size, -1])

else:
    raise ValueError("Invalid latent vector treatment", FLAGS.latent)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    encoder.load_weights(os.path.join(FLAGS.encoder, "encoder.h5"))

    with open(FLAGS.out_file, "wb") as f:
        # first int32 is number of lines, second i32 is dimensionality
        f.write(np.array([0, 0], dtype=np.int32).tobytes())

        try:
            count, dims = 0, None
            while True:
                c = sess.run(codes)
                b, d = c.shape

                if dims is not None:
                    assert dims == d
                else:
                    dims = d

                f.write(c.astype(np.float32).ravel().tobytes())
                count += b
                print(count, flush=True)

        except tf.errors.OutOfRangeError:
            f.seek(0)
            f.write(np.array([count, dims], dtype=np.int32).tobytes())
            print("Finished. count", count, "dims", dims)
