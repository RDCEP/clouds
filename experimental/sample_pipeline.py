import sys
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from reproduction.pipeline.load import add_pipeline_cli_arguments, load_data


def get_args():
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Saves samples from the given dataset pipeline.",
    )
    p.add_argument("output_dir")
    p.add_argument("--n_samples", type=int, default=64)
    add_pipeline_cli_arguments(p)
    return p.parse_args()


def sample_dataset(dataset, n):
    batch = dataset.make_one_shot_iterator().get_next()
    samples = []
    with tf.Session() as sess:
        while len(samples) < n:
            names, coords, imgs = sess.run(batch)
            samples.extend(list(zip(names, zip(*coords), imgs)))
    samples = samples[:n]
    return samples


def plot_samples(samples, fields, width=3):
    fig, ax = plt.subplots(
        nrows=len(samples),
        ncols=len(fields),
        figsize=(len(fields) * width, len(samples) * width),
    )
    for row, (f, coord, img) in enumerate(samples):
        for col, field in enumerate(fields):
            a = ax[row, col]
            a.imshow(img[:, :, col], cmap="bone")
            # a.set_axis_off()
            a.set_xticks([])
            a.set_yticks([])
            if row == 0:
                a.set_title(field)
            if row and col == 2:
                name = os.path.basename(str(f))
                name = os.path.splitext(name)[0]
                a.set_title("{}:{}".format(name, coord), fontsize=10)
    return fig


def plot_hists(samples, fields):
    samples = np.stack([img for _, _, img in samples])
    fig, ax = plt.subplots(ncols=len(fields), figsize=(len(fields) * 5, 5))

    for i, f in enumerate(fields):
        a = ax[i]
        a.hist(samples[:, :, :, i].ravel(), density=True)
        a.set_title(f)
    return fig


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

    sample_dataset(dataset, FLAGS.n_samples)

    # Save figs
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    fig = plot_samples(samples, FLAGS.fields)
    fig.savefig(os.path.join(FLAGS.output_dir, "samples.png"))

    fig = plot_hists(samples, FLAGS.fields)
    fig.savefig(os.path.join(FLAGS.output_dir, "hists.png"))
