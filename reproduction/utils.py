"""Common utilities for loading models, logging flags, and what not.
"""
__author__ = "casperneo@uchicago.edu"
import logging
import subprocess
import numpy as np
import tensorflow as tf
from os import path, listdir


def log_flag_arguments(flags, tf_version=True):
    """Logs the git commit, flags, and tensorflow version.
    """
    commit = subprocess.check_output(["git", "describe", "--always"]).strip()
    if tf_version:
        logging.info("Tensorflow version: %s", tf.__version__)
    logging.info("Current Git Commit: %s", commit)
    logging.info("Flags:")
    for f in flags.__dict__:
        logging.info("\t %s" + " " * (25 - len(f)) + "%s", f, flags.__dict__[f])


def load_encodings(encodings):
    """Loads binary of encoded patches from `encodings` and returns them as np.ndarray.
    Args:
        encodings: path to a file containing the encodings. The first 8 bytes are two
            int32 which specify the number of the encoding vectors and their dimension.
            The rest of the file are the float32 elements of these vectors.
    Returns:
        ndarray
    """
    with open(encodings, "r") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=-1).reshape((n, d))
    return data


def load_model_def(model_dir, name, weights=False):
    """Loads the model definition `name`.json from `model_dir`.
    """
    json = path.join(model_dir, name + ".json")
    if path.exists(json):
        with open(json, "r") as f:
            model = tf.keras.models.model_from_json(f.read())
        logging.info("model definition loaded from", json)
        if weights:
            load_latest_model_weights(model, model_dir, name)
        return model
    return None


def load_latest_model_weights(model, model_dir, name):
    """Loads the latest weights for `model.
    Args:
        model: Keras model
        model_dir: path to where model definition and weights are saved
        name: string of what the model is called.

    Returns:
        Number of steps which the loaded weights have been trained.
        Side effect: loads these weights into `model`

    Model weights are assumed to be saved in the format:
        `model_dir`/`name`-`step`.h5
    """
    latest = 0, None
    for m in listdir(model_dir):
        if ".h5" in m and name in m:
            step = int(m.split("-")[1].replace(".h5", ""))
            latest = max(latest, (step, m))
    step, model_file = latest

    if model_file:
        model_file = path.join(model_dir, model_file)
        model.load_weights(model_file)
        logging.info("loaded weights for %s from %s", name, model_file)

    else:
        logging.info("no weights for %s in %s", name, model_dir)

    return step
