"""Scratch work trying to use `sporco` for sparse dictionary learning.
"""
__author__ = "casperneo@uchicago.edu"


import numpy as np
from sporco.dictlrn import bpdndl
from sporco.util import grid_search

ENCODINGS = "output/mod09cnn15b/m15b-enc"

with open(ENCODINGS, "r") as f:
    n, d = np.fromfile(f, dtype=np.int32, count=2)
    data = np.fromfile(f, dtype=np.float32, count=n * d).reshape((n, d))[3000:].T

sparse_dict = np.random.randn(d, d * 4)

lmbda = 0.1
opt = bpdndl.BPDNDictLearn.Options(
    {
        'Verbose': True,
        'MaxMainIter': 100,
        'BPDN': {'rho': 10.0 * lmbda + 0.1},
        'CMOD': {'rho': data.shape[1] / 1e3}
    }
)
d = bpdndl.BPDNDictLearn(sparse_dict, data, lmbda, opt)
solution = d.solve()
print("BPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))

import IPython
IPython.embed()
#
#
# opt = ElasticNet.Options(
#     {
#         "Verbose": True,
#         "MaxMainIter": 500,
#         "RelStopTol": 1e-3,
#         "AutoRho": {"RsdlTarget": 1.0},
#     }
# )
#
#
# original = data[:1000].T
# l1weight, l2weight = 10.0, 10.0
# b = ElasticNet(sparse_dict, original, l1weight, l2weight, opt)
# reconstruction = b.solve()
# assert reconstruction.shape == (128, 1000), "R000" + str(reconstruction.shape)
#
#
# ###############################################################################
# #   Grid search                                                               #
# ###############################################################################
#
#
# def error(reg):
#     print(reg)
#     l1weight, l2weight = reg
#     original = data[:1000].T
#
#     assert original.shape == (128, 1000), "O"
#
#     b = ElasticNet(sparse_dict, original, l1weight, l2weight, opt)
#     reconstruction = b.solve()
#
#     assert reconstruction.shape == (128, 1000), "R" + str(reconstruction.shape)
#
#     return np.sum(np.abs(reconstruction - original))
#
#
# regularizations = np.logspace(1, 2, 5), np.logspace(1, 2, 5)
# sprm, sfvl, fvmx, sidx = grid_search(error, regularizations, nproc=1)
#
# print("sprm", sprm)
# print("sfvl", sfvl)
# print("fvmx", fvmx)
# print("sidx", sidx)
#
#
# x = e.solve()
# print(x)
