"""Train sparse dictionary with sporco.
"""
__author__ = "casperneo@uchicago.edu"
import os
import numpy as np
from argparse import ArgumentParser
from sporco.dictlrn import bpdndl
from sporco.util import grid_search
from sporco import plot

p = ArgumentParser(description=__doc__)
p.add_argument(
    "encodings",
    help="File containing encodings. Formatted as a binary where the file starts with "
    "two int32s corresponding to number of vectors `n` and their dimensionality `d`. The "
    "rest of the file should be float32s corresponding to the vector elements.",
)
p.add_argument("output", help="directory to save trained dictionary and training figures")
p.add_argument("--num_iterations", default=1000, type=int)

FLAGS = p.parse_args()
os.makedirs(FLAGS.output, exist_ok=True)


# Get encodings to train on
with open(FLAGS.encodings, "r") as f:
    n, d = np.fromfile(f, dtype=np.int32, count=2)
    data = np.fromfile(f, dtype=np.float32, count=n * d).reshape((n, d))


# Train sparse dictionary
lmbda = 0.1
opt = bpdndl.BPDNDictLearn.Options(
    {
        "Verbose": True,
        "MaxMainIter": FLAGS.num_iterations,
        "BPDN": {"rho": 10.0 * lmbda + 0.1},
        "CMOD": {"rho": n / 1e3},
    }
)
D0 = np.random.randn(d, 2 * d)
d = bpdndl.BPDNDictLearn(D0, data[:10000].T, lmbda, opt)
D1 = d.solve()
np.save(os.path.join(FLAGS.output, "trained_dictionary.npy"), D1)


# Plot iteration statistics
its = d.getitstat()
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

fig.savefig(os.path.join(FLAGS.output, "train-stats.png"))
print("BPDNDictLearn solve time: %.2fs" % d.timer.elapsed("solve"))
