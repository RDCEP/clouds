import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.cluster.vq import kmeans
import tensorflow as tf
from dnn import pipeline

import matplotlib

#matplotlib.use("agg")  # This avoids RuntimeError Invalid DISPLAY variable
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_ae_output(dataset, predictions, n_samples, n_bands, height=4, width=4):
    """
    """
    fig, ax = plt.subplots(
        figsize=(n_bands * width, n_samples * 2 * height),
        nrows=n_samples * 2,
        ncols=n_bands,
    )

    for i in range(n_samples):
        for j in range(n_bands):
            orig, pred = ax[i * 2 : i * 2 + 2, j]
            orig.imshow(dataset[i, :, :, j], cmap="copper")
            pred.imshow(predictions[i, :, :, j], cmap="copper")
            for a in (orig, pred):
                a.set_xlabel("band %d" % (j + 1))
                a.axis("off")

    return fig


class PCA:
    def __init__(self, vectors):
        """Find eigenvalues and vectors
        """
        self.mean = vectors.mean(axis=0)
        centered = vectors - self.mean
        cov = centered.transpose().dot(centered) / centered.shape[0]
        evals, evecs = np.linalg.eigh(cov)
        # Remove useless axis
        gz = evals > 0.1
        evals = evals[gz]
        evecs = evecs[gz]
        print(evals)

        # So PCA projection also whitens data for viewing
        for i in range(evals.shape[0]):
            evecs[:, i] /= evals[i]

        self.evals = evals
        self.evecs = evecs

    def project(self, vectors, dim):
        """Project centered vectors onto leading `dim` eigenvectors
        """
        centered = vectors - self.mean
        return centered.dot(self.evecs[-dim:].transpose())


def img_scatter(points, images, zoom=0.5):
    """Scatter plot where points are overlaid with images
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    for ((x, y), img) in zip(points, images):
        im = OffsetImage(img, zoom=zoom, cmap="copper")
        ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
        ax.add_artist(ab)
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    return fig, ax


def plot_kmeans_and_image_scatter(original, encoded, K=3):
    """Plots AE encoded space projected down by PCA with scattered images and
    k means.
    """
    # Spatial average features
    centered = encoded.mean(axis=(1, 2))
    centered = centered - centered.mean(axis=0)
    pc = PCA(centered)
    proj = pc.project(centered, 2)

    # Average over all the bands for plotting purposes
    imgs = original.mean(axis=3)
    fig, ax = img_scatter(proj, imgs, zoom=0.5)
    ax.set_title("Autoencoder hidden space PCA projection")

    # Add K Means cluster centers projected down
    codebook, distortion = kmeans(centered, K)
    xs, ys = pc.project(codebook, 2).transpose()
    ax.scatter(xs, ys, s=100, c="r", zorder=1000)

    return fig, ax


if __name__ == "__main__":
    import argparse

    # TODO Dont repeat common loading stuff in this section and in train.py
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("data", help="/path/to/data.tif")
    p.add_argument(
        "-sh",
        "--shape",
        nargs=3,
        type=int,
        help="Shape of input image",
        default=(64, 64, 7),
    )
    p.add_argument(
        "model_dir", help="/path/to/model/ to load model from and to save inference"
    )
    p.add_argument("-ns", "--num_samples", type=int, help=" ", default=1000)

    FLAGS = p.parse_args()
    if FLAGS.model_dir[-1] != "/":
        FLAGS.model_dir += "/"

    # TODO path join
    with open(FLAGS.model_dir + "ae.json", "r") as f:
        ae = tf.keras.models.model_from_json(f.read())
    ae.load_weights(FLAGS.model_dir + "ae.h5")

    img_width, img_height, n_bands = FLAGS.shape
    del img_height  # Unused TODO maybe use it?
    x = (
        tf.data.Dataset.from_generator(
            pipeline.read_tiff_gen([FLAGS.data], img_width),
            tf.float32,
            (img_width, img_width, n_bands),
        )
        .apply(tf.contrib.data.shuffle_and_repeat(100))
        .batch(FLAGS.num_samples)
        .make_one_shot_iterator()
        .get_next()
    )

    with tf.Session() as sess:
        x = sess.run(x)

    # TODO how to pick hidden layer?
    # en = tf.keras.models.Model(inputs=ae.input, outputs=ae.get_layer("conv2d_3").output)
    # e = en.predict(x)
    # y = ae.predict(x)[0]
    e, y = ae.predict(x)

    # from IPython import embed
    # embed()

    # Save autoencoder output
    plot_ae_output(x, y, 2, n_bands)
    fname = FLAGS.model_dir + "ae_output.png"
    plt.savefig(fname)
    print(f"Autoencoder Results saved to {fname}")

    # Save Hidden space analysis
    fig, _ = plot_kmeans_and_image_scatter(x, e)
    fname = FLAGS.model_dir + "pca.png"
    fig.savefig(fname)
    print(f"Hidden space diagram saved to {fname}")
