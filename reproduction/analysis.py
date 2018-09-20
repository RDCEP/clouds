import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.cluster.vq import kmeans
import tensorflow as tf
import matplotlib

# matplotlib.use("agg")  # This avoids RuntimeError Invalid DISPLAY variable
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class AEData:
    def __init__(self, dataset, ae, fields, n=500):
        names, coords, imgs = zip(*sample_dataset(dataset, n))
        imgs = np.stack(imgs)
        encs, ae_imgs = ae.predict(imgs)

        self.names = [str(n) for n in names]
        self.coords = coords
        self.imgs = imgs
        self.ae_imgs = ae_imgs
        self.encs = encs.mean(axis=(1, 2))
        self.raw_encs = encs
        self.fields = fields

        centered = self.encs - self.encs.mean(axis=0)
        cov = centered.transpose().dot(centered) / centered.shape[0]
        evals, evecs = np.linalg.eigh(cov)
        evals = np.flip(evals)
        evecs = np.flip(evecs, axis=1)
        self._evals = evals
        self._evecs = evecs


    def pca_project(self, x, d=3):
        centered = x.encs - self._evals
        if isinstance(d, list):
            return centered.dot(self._evecs[:, d]).transpose()
        else:
            return centered.dot(self._evecs[:, :d]).transpose()

    def plot_pca_projection(self, x, title="", width=3, c=None):
        pc = self.pca_project(x)
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(width * 3 + 2, width))
        for i in range(3):
            j = (i + 1) % 3
            a = ax[i]
            im = a.scatter(pc[i], pc[j], c=c)
            a.set_xlabel("PC %d"%i)
            a.set_ylabel("PC %d"%j)

        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.99, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        fig.suptitle(title)

    def plot_residuals(self, n_samples=2, width=2):
        fig, ax = plt.subplots(
            nrows=n_samples * 3,
            ncols=len(self.fields),
            figsize=(len(self.fields) * width, n_samples * width * 3),
        )
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        for s in range(n_samples):
            for c, field in enumerate(self.fields):
                orig, diff, deco = ax[s * 3 : s * 3 + 3, c]
                orig.imshow(self.imgs[s, :, :, c], cmap="bone")
                diff.imshow(
                    self.imgs[s, :, :, c] - self.ae_imgs[s, :, :, c], cmap="coolwarm"
                )
                deco.imshow(self.ae_imgs[s, :, :, c], cmap="bone")
                if s == 0:
                    orig.set_title(field)
                if c == 0:
                    orig.set_ylabel("original")
                    diff.set_ylabel("residual")
                    deco.set_ylabel("decoded")

        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
        return fig, ax


def sample_dataset(dataset, n):
    batch = dataset.make_one_shot_iterator().get_next()
    samples = []
    with tf.Session() as sess:
        while len(samples) < n:
            names, coords, imgs = sess.run(batch)
            samples.extend(zip(names, coords, imgs))
    samples = samples[:n]
    return samples


def plot_cluster_channel_distributions(imgs, labels, fields=None, width=3):
    """Plot histograms of channel values for each cluster.
    """
    n_bands = imgs.shape[-1]
    assert (
        not fields or len(fields) == n_bands
    ), "Number of field labels do not match number of channels"
    n_clusters = len(set(labels))

    fig, ax = plt.subplots(
        nrows=n_bands, ncols=n_clusters, figsize=(n_clusters * width, n_bands * width)
    )

    for i in range(n_bands):
        for j in range(n_clusters):
            a = ax[i, j]
            a.hist(imgs[labels == j, :, :, i].ravel())
            a.set_yticks([])
            a.set_xlim(left=imgs[:, :, :, i].min(), right=imgs[:, :, :, i].max())

    for i in range(n_bands):
        ax[i, 0].set_ylabel(fields[i] if fields else "band %d" % i)

    for j in range(n_clusters):
        ax[0, j].set_title("cluster %d" % j)

    return fig, ax


def plot_cluster_samples(imgs, labels, samples=8, width=3, channel=0):
    n_clusters = len(set(labels))

    fig, ax = plt.subplots(
        nrows=samples, ncols=n_clusters, figsize=(n_clusters * width, samples * width)
    )
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for i in range(n_clusters):
        n = (labels == i).sum()
        ax[0, i].set_title("cluster %d" % i)

        for j, k in enumerate(np.random.choice(n, samples, replace=False)):
            a = ax[j, i]
            img = imgs[labels == i]
            a.imshow(img[k, :, :, channel], cmap="bone")
            a.set_yticks([])
            a.set_xticks([])

    return fig, ax


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


def img_scatter(points, images, zoom=0.5, figsize=(20, 20)):
    """Scatter plot where points are overlaid with images
    """
    fig, ax = plt.subplots(figsize=figsize)

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


# TODO depricate most of this main code
if __name__ == "__main__":
    import argparse
    import pipeline

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
