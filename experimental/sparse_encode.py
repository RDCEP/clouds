"""Scratch work trying to use `sporco` for sparse dictionary learning.
"""
__author__ = "casperneo@uchicago.edu"


import numpy as np
from sporco.admm.bpdn import ElasticNet
from sporco.util import grid_search

ENCODINGS = "output/mod09cnn15b/m15b-enc"

with open(ENCODINGS, "r") as f:
    n, d = np.fromfile(f, dtype=np.int32, count=2)
    data = np.fromfile(f, dtype=np.float32, count=n * d).reshape((n, d))

print(data.shape)

sparse_dict = np.random.random((d, d * 4))

import IPython

IPython.embed()


opt = ElasticNet.Options(
    {
        "Verbose": True,
        "MaxMainIter": 500,
        "RelStopTol": 1e-3,
        "AutoRho": {"RsdlTarget": 1.0},
    }
)


original = data[:1000].T
l1weight, l2weight = 10.0, 10.0
b = ElasticNet(sparse_dict, original, l1weight, l2weight, opt)
reconstruction = b.solve()
assert reconstruction.shape == (128, 1000), "R000" + str(reconstruction.shape)


###############################################################################
#   Grid search                                                               #
###############################################################################


def error(reg):
    print(reg)
    l1weight, l2weight = reg
    original = data[:1000].T

    assert original.shape == (128, 1000), "O"

    b = ElasticNet(sparse_dict, original, l1weight, l2weight, opt)
    reconstruction = b.solve()

    assert reconstruction.shape == (128, 1000), "R" + str(reconstruction.shape)

    return np.sum(np.abs(reconstruction - original))


regularizations = np.logspace(1, 2, 5), np.logspace(1, 2, 5)
sprm, sfvl, fvmx, sidx = grid_search(error, regularizations, nproc=1)

print("sprm", sprm)
print("sfvl", sfvl)
print("fvmx", fvmx)
print("sidx", sidx)


x = e.solve()
print(x)
