"""Opens a file of vectors formatted for `parallel-kmeans` and saves a permutation.

`parallel-kmeans` is the library we're using to scale k-means clustering. The file format
is a binary with two int32s representing the number of vectors and their dimensionality.
The rest of the file are float32s representing the values of the vectors in C ordering.
This program will open this file, permute the order of the vectors and and save it to
another file.
"""
__author__ = "casperneo@uchicago.edu"

import numpy as np
from argparse import ArgumentParser

p = ArgumentParser(description=__doc__)
p.add_argument("vectors", help="vectors to premute")
p.add_argument("permuted", help="file to keep permuted vectors")
FLAGS = p.parse_args()


with open(FLAGS.vectors) as f:
    n, d = np.fromfile(f, np.int32, 2)
    vectors = np.fromfile(f, np.float32, -1).reshape((n, d))

perm = np.random.permutation(n)

with open(FLAGS.permuted, "w") as f:
    f.write(n)
    f.write(d)
    f.write(data[perm].tobytes())

with open(FLAGS.permuted + ".permutation", "w") as f:
    perm.tofile(f, sep=" ")
