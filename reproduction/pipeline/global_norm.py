"""Experimental code that tries to find a global mean and stdev so patches could be
normalized on the fly when being read.
"""

__author__ = "casperneo@uchicago.edu"

import os
import json
import glob
import numpy as np

# from osgeo import gdal
from mpi4py import MPI
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pyspark import SparkContext
from pyspark.mllib.stat import Statistics


MOD02_FIELDS = (
    "EV_250_Aggr1km_RefSB",
    "EV_500_Aggr1km_RefSB",
    "EV_1KM_RefSB",
    "EV_1KM_Emissive",
)

if __name__ == "__main__":
    # python's importing system is weird. When importing this file from another directory
    # into_record doesn't exist
    from into_record import read_hdf

    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            "Computes global statistics for each channel in every image for normalization."
            "Results are stored a json.",
        ),
    )
    p.add_argument("source_glob", help="Glob of files to compute normalization from")
    p.add_argument("out_json", help="File to save results")
    p.add_argument(
        "--fields", nargs="+", help="Required if filetype is hdf", default=MOD02_FIELDS
    )

    FLAGS = p.parse_args()
    FLAGS.out_json = os.path.abspath(FLAGS.out_json)

    for f in FLAGS.__dict__:
        print("\t", f, (25 - len(f)) * " ", FLAGS.__dict__[f])
    print("\n", flush=True)

    def pixels(f):
        swath = read_hdf(f, FLAGS.fields)
        swath = np.rollaxis(swath, 0, 3)
        return swath.reshape((-1, swath.shape[2]))

    sc = SparkContext()
    sc.addPyFile(os.path.abspath("reproduction/pipeline/into_record.py"))

    summary = Statistics.colStats(
        sc.parallelize(glob.glob(FLAGS.source_glob)).flatMap(pixels)
    )

    mean = summary.mean()
    std = np.sqrt(summary.variance())

    j = {"sub": list(mean), "div": list(std)}

    with open(FLAGS.out_json, "w") as f:
        json.dump(j, f)
