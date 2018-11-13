"""Run experiments on clustering algorithms to measure consistency.

Consistency can be defined as high adjusted mutual information in cluster assignments
under perturbations such as, reinitialization, adding more data, adding another cluster,
or changing the neural network encoding the data.
"""
__author__ = "casperneo@uchicago.edu"

from sklearn import cluster
