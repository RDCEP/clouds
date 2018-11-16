"""Sparse coding utilities. The main class is `SparseCoder`.
"""
__author__ = "casperneo@uchicago.edu"
import os
import logging
import numpy as np
from sporco import plot
from sporco.admm import bpdn
from sporco.dictlrn import bpdndl
from argparse import ArgumentParser
from utils import load_encodings, load_model_def, load_latest_model_weights


class SparseCoder:
    """Holds and trains a sparse dictionary alongside its associated encoder and decoder.
    """

    def __init__(
        self, sparse_dict_path, nn_path, code_mult=0.5, spatial_average=False, l1_reg=10
    ):
        """
        Args:
            sparse_dict_path: Path to saved sparse dictionary, or where to save it.
            nn_path: Path to trained AE encoder and decoder models.
            code_mult: Number of codewords to use in a new dict as multiple of code dim
            spatial_average: Reduce code dim by averaging over spatial dimensions
            l1_reg: Sparsity coefficient for sparse coding optimization
        """
        self.sd_path = sparse_dict_path
        self.nn_path = nn_path
        self.l1_reg = l1_reg
        self.spatial_average = spatial_average

        self.encoder = load_model_def(nn_path, "encoder")
        self.decoder = load_model_def(nn_path, "decoder")
        self.step = load_latest_model_weights(self.encoder, nn_path, "encoder")
        load_latest_model_weights(self.decoder, nn_path, "decoder")

        if spatial_average:
            code_dim = self.encoder.output_shape[3]
        else:
            code_dim = np.product(self.encoder.output_shape[1:])

        num_codes = int(code_mult * code_dim)

        if os.path.exists(sparse_dict_path):
            self.dict = np.fromfile(sparse_dict_path)
            logging.info("Sparse dictionary loaded from `%s`", self.dict)
            if n_codes is not None and dim is not None:
                logging.warning(
                    "`n_codes={}`, `dim={}`".format(n_codes, code_dim)
                    + "ignored as sparse_dict_path loaded from file."
                )
        else:
            self.dict = np.random.randn(code_dim, num_codes)
            logging.info("Sparse dictionary initialized randomly")

    def train_dict(self, encodings, num_steps=100):
        """Trains `self.dict` on `encodings` using `sporco.admm.bpdndl` for `num_steps`.
        Args:
            encodings: ndarray with shape (code_dim, num_examples)
            num_steps: Number of bpdndl main iteration steps
        """
        opt = bpdndl.BPDNDictLearn.Options(
            {
                "Verbose": True,
                "MaxMainIter": num_steps,
                "BPDN": {"rho": 10.0 * self.l1_reg + 0.1},
                "CMOD": {"rho": encodings.shape[1] / 1e3},
            }
        )
        # Basis Pursuit DeNoising Dictionary Learning
        self.dictlearn = bpdndl.BPDNDictLearn(self.dict, encodings, self.l1_reg, opt)
        logging.info("Beginning training")
        self.dict = self.dictlearn.solve()
        logging.info(
            "BPDNDictLearn solve time: %.2fs", self.dictlearn.timer.elapsed("solve")
        )

        np.save(self.sd_path, self.dict)

    def evaluate_images(self, images, num_iterations=500):
        """Returns AE reconsturction loss and sparse-coded reconstruction loss for images.
        """
        if self.spatial_average:
            raise ValueError(
                "reconsturction loss undefined for spatially averge sparse coding"
            )

        latent_vectors = self.encoder.predict(images).reshape((images.shape[0], -1))

        opt = bpdn.BPDN.Options(
            {
                "Verbose": False,
                "MaxMainIter": num_iterations,
                "RelStopTol": 1e-3,
                "AutoRho": {"RsdlTarget": 1.0},
            }
        )
        b = bpdn.BPDN(self.dict, latent_vectors, self.l1_reg, opt)
        codes = b.solve()
        raise NotImplementedError("TODO")

    def save_train_stats(self, path=None):
        """Save dictionary learning training statistics as per
        https://sporco.readthedocs.io/en/latest/examples/dl/bpdndl.html#example-dl-bpdndl
        """
        its = self.dictlearn.getitstat()
        fig = plot.figure(figsize=(20, 5))

        plot.subplot(1, 3, 1)
        plot.plot(its.ObjFun, xlbl="Iterations", ylbl="Functional", fig=fig)

        plot.subplot(1, 3, 2)
        plot.plot(
            np.vstack((its.XPrRsdl, its.XDlRsdl, its.DPrRsdl, its.DDlRsdl)).T,
            ptyp="semilogy",
            xlbl="Iterations",
            ylbl="Residual",
            lgnd=["X Primal", "X Dual", "D Primal", "D Dual"],
            fig=fig,
        )

        plot.subplot(1, 3, 3)
        plot.plot(
            np.vstack((its.XRho, its.DRho)).T,
            xlbl="Iterations",
            ylbl="Penalty Parameter",
            ptyp="semilogy",
            lgnd=["$\\rho_X$", "$\\rho_D$"],
            fig=fig,
        )

        if path is None:
            path = self.sd_path.replace(".npy", "") + "-train-stats.png"
        fig.savefig(path)


if __name__ == "__main__":
    # DEBUG this is testing code
    logging.basicConfig(level=logging.DEBUG)
    sc = SparseCoder("foob", "output/mod09cnn15b")
    encs = load_encodings("encs")
    sc.train_dict(encs.T)
    sc.save_train_stats()
