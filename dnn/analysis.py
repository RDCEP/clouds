import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.cluster.vq import kmeans

import matplotlib

matplotlib.use("agg")  # This avoids RuntimeError Invalid DISPLAY variable
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
            orig.imshow(dataset[i, :, :, j], cmap="binary")
            pred.imshow(predictions[i, :, :, j], cmap="binary")
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
        self.evals, self.evecs = np.linalg.eigh(cov)

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
        im = OffsetImage(img, zoom=zoom, cmap="binary")
        ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
        ax.add_artist(ab)
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    return fig, ax


def plot_kmeans_and_image_catter(original, encoded, K=10):
    """Plots AE encoded space projected down by PCA with scattered images and
    k means.
    """
    # Spatial average features
    centered = encoded - encoded.mean(axis=(1, 2))
    pc = PCA(centered)
    proj = pc.project(centered, 2)

    # Average over all the bands for plotting purposes
    fig, ax = analysis.img_scatter(proj, original.mean(axis=3), zoom=0.5)

    # Add K Means cluster centers projected down
    codebook, distortion = kmeans(centered, K)
    xs, ys = pc.project(codebook, 2).transpose()
    ax.scatter(xs, ys, s=100, c="r", zorder=1000)

    return fig, ax
