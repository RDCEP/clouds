import sys
sys.path.insert(0, "..")
import tensorflow as tf
from reproduction.pipeline.load import add_pipeline_cli_arguments, load_data
import os
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_args():
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Saves samples from the given dataset pipeline.",
    )
    p.add_argument("output_dir")
    p.add_argument("--n_samples", type=int, default=64)
    add_pipeline_cli_arguments(p)
    return p.parse_args()


def save_samples(fields, samples, fname, width=3):
    fig, ax = plt.subplots(
        nrows=len(samples),
        ncols=len(fields),
        figsize=(len(fields) * width, len(samples) * width),
    )
    for row, (f, coord, img) in enumerate(samples):
        for col, field in enumerate(fields):
            a = ax[row, col]
            a.imshow(img[:, :, col], cmap="bone")
            a.set_axis_off()
            if row == 0:
                a.set_title(field)
            if row and col == 0:
                a.set_title("{}:{}".format(f, coord))

    fig.savefig(fname)


def save_hists(fields, samples, fname):
    samples = np.stack([img for _, _, img in samples])
    fig, ax = plt.subplots(ncols=len(fields), figsize=(len(fields) * 5, 5))

    for i, f in enumerate(fields):
        a = ax[i]
        a.hist(samples[:, :, :, i].ravel(), density=True)
        a.set_title(f)
    fig.savefig(fname)


if __name__ == "__main__":
    FLAGS = get_args()

    chans, dataset = load_data(
        FLAGS.data,
        FLAGS.fields,
        FLAGS.meta_json,
        FLAGS.shape,
        FLAGS.batch_size,
        FLAGS.normalization,
        FLAGS.read_threads,
        FLAGS.prefetch,
        FLAGS.shuffle_buffer_size,
    )
    shape = (*FLAGS.shape, chans)

    batch = dataset.make_one_shot_iterator().get_next()

    # Get samples
    samples = []
    with tf.Session() as sess:
        while len(samples) < FLAGS.n_samples:
            names, coords, imgs = sess.run(batch)
            samples.extend(list(zip(names, zip(*coords), imgs)))
    samples = samples[: FLAGS.n_samples]

    # Save figs
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    save_samples(FLAGS.fields, samples, os.path.join(FLAGS.output_dir, "samples.png"))
    save_hists(FLAGS.fields, samples, os.path.join(FLAGS.output_dir, "hists.png"))
