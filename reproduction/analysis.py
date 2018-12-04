"""analysis.py: A Library for analysing and visualizing image-autoencoder hidden space.
This is mostly experimental and oneoff code that has been moved from the analysis
notebooks to keep them clean. The analysis notebooks import from this library.
"""
__author__ = "casperneo@uchicago.edu"
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import subprocess
import json
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.cluster.vq import kmeans
from collections import namedtuple
from matplotlib import patches
from osgeo import gdal

# TODO: Extend gdal exception control to the entire codebase
# Enable exception treatment using gdal
gdal.UseExceptions()


class AEData:
    """Struct of arrays containing autoencoded data for analysis.

    imgs        [n_samples, height, width, channels] array of patches
    names       tif-files where the patches are sourced from
    coords      index within tif file of the top left corner of the original patch
    ae_imgs     patches after being autoencoded
    raw_encs    encoded state of the patches
    encs        spatially averaged encoded state of the patches
    fields      names for each of the channels in imgs
    """

    def __init__(self, dataset, ae=None, fields=None, n=500):
        # get data from dataset
        names, coords, imgs = [], [], []
        batch = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            while len(imgs) < n:
                names_, coords_, imgs_ = sess.run(batch)
                names.extend(names_)
                coords.extend(coords_)
                imgs.extend(imgs_)

        self.names = np.array([str(n)[2:-1] for n in names])
        self.coords, self.imgs = np.stack(coords[:n]), np.stack(imgs[:n])
        self.colored_imgs = cmap_and_normalize(self.imgs)
        self.fields = (
            fields if fields else ["b%d" % (i + 1) for i in range(self.imgs.shape[-1])]
        )

        if ae is not None:
            self.compute_ae(ae)
            self.compute_pca()

    def add_encoder(self, encoder):
        self.encoder = encoder
        self.raw_encs = encoder.predict(self.imgs)
        self.encs = self.raw_encs.mean(axis=(1, 2))

    def compute_ae(self, ae):
        self.raw_encs, self.ae_imgs = ae.predict(self.imgs)
        self.encs = self.raw_encs.mean(axis=(1, 2))
        self.ae = ae

    def compute_pca(self):
        if not hasattr(self, "encs"):
            raise ValueError("Need to have encoded vectors to compute pca")

        centered = self.encs - self.encs.mean(axis=0)
        cov = centered.transpose().dot(centered) / centered.shape[0]
        evals, evecs = np.linalg.eigh(cov)
        evals = np.flip(evals)
        evecs = np.flip(evecs, axis=1)
        self._evals = evals
        self._evecs = evecs
        self._center = self.encs.mean(axis=0)

    def pca_project(self, x, d=3):
        if not hasattr(self, "_evecs"):
            self.compute_pca()
        centered = x - self._center
        if isinstance(d, list):
            return centered.dot(self._evecs[:, d]).transpose()
        else:
            return centered.dot(self._evecs[:, :d]).transpose()

    def plot_pca_projection(self, x, title="", width=3, cbar=True, **kwargs):
        pc = self.pca_project(x)
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(width * 3 + 2, width))
        for i in range(3):
            j = (i + 1) % 3
            a = ax[i]
            im = a.scatter(pc[i], pc[j], **kwargs)
            a.set_xlabel("PC %d" % i)
            a.set_ylabel("PC %d" % j)

        if cbar:
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
                orig, diff, deco = ax[s * 3: s * 3 + 3, c]
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

    def plot_neighborhood(self, i, context_width=128):
        p, orig = self.open_neighborhood(i, context_width)

        _, (a, b) = plt.subplots(1, 2, figsize=(20, 10))
        normalization = None  # plt.Normalize(p[0].min(), p[0].max())
        a.imshow(self.imgs[i, :, :, 0], norm=normalization, cmap="bone")
        b.imshow(p[0], norm=normalization, cmap="bone")
        b.add_patch(
            patches.Rectangle(
                orig,
                *self.imgs.shape[1:3],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )

    def open_neighborhood(self, i, context_width, override_base_folder=None):
        """Opens `context_width` size neighborhood around patch `i`.
        Returns this enlarged patch (unnormalized) and the coordinate of the original
        patch within it.
        """
        yoff, xoff = self.coords[i]
        xsize, ysize = self.imgs.shape[1:3]

        def rebox(off, size, mx):
            l_most = max(off - context_width, 0)
            r_most = min(mx, off + size + context_width)
            new_size = r_most - l_most
            return map(int, [l_most, new_size, off - l_most])

        swath = None  # Initialize variable due to exception treatment
        # TODO: Provide a better fix for the issue of source file url used on training
        # Issue lies on full url names being stored, instead of relative paths
        if not (override_base_folder is None):
            tif_filename = os.path.basename(self.names[i])
            newurl = override_base_folder+tif_filename
            print('WARNING: Overriding base folder name, from the one used on model training', flush=True)
        else:
            newurl = self.names[i]
        try:
            swath = gdal.Open(newurl)
        except Exception as e:
            print('ERROR:', e.message, e.args, flush=True)

        xoff, xsize, left = rebox(xoff, xsize, swath.RasterXSize)
        yoff, ysize, top = rebox(yoff, ysize, swath.RasterYSize)

        p = swath.ReadAsArray(xoff, yoff, xsize, ysize)

        return p, (left, top)


def sample_dataset(dataset, n):
    batch = dataset.make_one_shot_iterator().get_next()
    samples = []
    with tf.Session() as sess:
        while len(samples) < n:
            names, coords, imgs = sess.run(batch)
            samples.extend(zip(names, coords, imgs))
    samples = samples[:n]
    return samples


def cmap_and_normalize(imgs, reds=[1, 4, 5, 6], greens=[0], blues=[2, 3], maxper=95):
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, 0)

    colors = []
    for col in (reds, greens, blues):
        ii = imgs[:, :, :, col].mean(axis=3)
        ii = np.clip(ii, *np.percentile(ii, [0, maxper]))
        ii /= ii.max() - ii.min()
        ii -= ii.min()
        colors.append(ii)

    return np.stack(colors, axis=3)


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


def plot_cluster_samples(imgs, labels, samples=8, width=3):
    n_clusters = len(set(labels))

    # TODO use 1 axis and manually do subplots because its faster
    fig, ax = plt.subplots(
        nrows=samples, ncols=n_clusters, figsize=(n_clusters * width, samples * width)
    )
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for a in ax.ravel():
        a.set_yticks([])
        a.set_xticks([])

    for i in range(n_clusters):
        n = (labels == i).sum()
        ax[0, i].set_title("cluster %d" % i)
        if n > samples:
            choices = enumerate(np.random.choice(n, samples, replace=False))
        else:
            choices = [(j, j) for j in range(n)]
        for j, k in choices:
            img = imgs[labels == i][k]
            if len(img.shape) > 2 and img.shape[2] != 3:
                img = img[:, :, 0]
            ax[j, i].imshow(img, cmap="bone")

    return fig, ax


def plot_all_cluster_samples(imgs, labels, order=None, samples=None, width=2):
    """Plots all examples in a cluster in 1 column, order determins which clusters and in
    what order are plotted.
    """
    uniq, counts = np.unique(labels, return_counts=True)
    ncols = len(uniq)
    nrows = max(counts)
    _, patch_height, patch_width = imgs.shape
    if order is None:
        order = np.argsort(counts)

    fig, ax = plt.subplots(1, ncols, figsize=(ncols * width, nrows * width))

    for idx, a in zip(order, ax):
        l = uniq[idx]
        count = counts[idx]
        column = imgs[labels == l].reshape((count * patch_height, patch_width))
        a.imshow(column, cmap="bone")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title("Cluster %d" % l)

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
            orig, pred = ax[i * 2: i * 2 + 2, j]
            orig.imshow(dataset[i, :, :, j], cmap="copper")
            pred.imshow(predictions[i, :, :, j], cmap="copper")
            for a in (orig, pred):
                a.set_xlabel("band %d" % (j + 1))
                a.axis("off")

    return fig


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
    """Plots AE encoded space projected down by PCA with scattered images and k means.
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


def _dict_to_named_tuple(name, d):
    """Recursively convert dictionary into a named tuple because its prettier.
    """
    if not isinstance(d, dict):
        return d

    if "" in d:
        d["none"] = d.pop("")

    values = [_dict_to_named_tuple(k, d[k]) for k in d]
    return namedtuple(name, d.keys())(*values)


def get_tif_metadata(tif_file, as_dict=False):
    s = subprocess.check_output(["gdalinfo", "-json", tif_file])
    j = json.loads(s)
    return j if as_dict else _dict_to_named_tuple("tif_metadata", j)


def read_kmeans_centers(filename, is_ascii=True):
    if not is_ascii:
        raise NotImplementedError()
    with open(filename, "r") as f:
        centers = [
            [float(x) for x in line.strip().split(" ")[1:]] for line in f.readlines()
        ]
    return np.array(centers)
