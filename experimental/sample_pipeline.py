import sys
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from IPython import embed  # DEBUG

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from reproduction.pipeline.load import add_pipeline_cli_arguments, load_data
from reproduction.analysis import sample_dataset


def get_args():
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Saves samples from the given dataset pipeline.",
    )
    p.add_argument("output_dir")
    p.add_argument("--n_samples", type=int, default=64)
    add_pipeline_cli_arguments(p)
    return p.parse_args()


def plot_samples(samples, fields, width=3):
    fig, ax = plt.subplots(
        nrows=len(samples) + 1,
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

    imgs = np.array([img for _, _, img in samples])
    for i, a in enumerate(ax[len(samples)]):
        a.hist(imgs[:, :, :, i].ravel())

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

    dataset = load_data(
        FLAGS.data,
        FLAGS.shape,
        FLAGS.batch_size,
        FLAGS.read_threads,
        FLAGS.shuffle_buffer_size,
        FLAGS.prefetch,
    )
    # HACK:
    FLAGS.fields = ["b%d" % (i + 1) for i in range(FLAGS.shape[2])]

    samples = sample_dataset(dataset, FLAGS.n_samples)

    # Save figs
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    fig = plot_samples(samples, FLAGS.fields)
    fig.savefig(os.path.join(FLAGS.output_dir, "samples.png"))
