from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
p.add_argument("model_dir")
p.add_argument("--tif_files", nargs="*", default=[])
p.add_argument("--hdf_files", nargs="*", default=[])
p.add_argument("--img_width", default=64)
p.add_argument("--n_bands", default=7)
p.add_argument("--batch_size", default=32)
p.add_argument("--use_hdf_data", action="store_true", default=False)
FLAGS = p.parse_args()

import pipeline
import os
import tensorflow as tf
from tensorflow.contrib.data import shuffle_and_repeat, batch_and_drop_remainder
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

### Load Data


if bool(FLAGS.tif_files) == bool(FLAGS.hdf_files):
    print("Must provied either --tif_files or --hdf_files")
    exit(1)

tif_fields = ["b%d" % (i + 1) for i in range(FLAGS.n_bands)]
tif_dataset = (
    tf.data.Dataset.from_generator(
        pipeline.read_tiff_gen(
            FLAGS.tif_files, FLAGS.img_width
        ),  # TODO: Shift for read_tiff_gen
        tf.int16,
        (FLAGS.img_width, FLAGS.img_width, FLAGS.n_bands),
    )
    .apply(tf.contrib.data.shuffle_and_repeat(100))
    .apply(batch_and_drop_remainder(FLAGS.batch_size))
)

if FLAGS.hdf_files:
    print("HDF not loaded...")
    exit(1)
    # dataset = hdf_dataset
    # fields = hdf_fields
else:
    dataset = tif_dataset
    fields = tif_fields

x = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    y = sess.run(x)

# -- Load DNN model

with open(FLAGS.model_dir + "ae.json", "r") as f:
    ae = tf.keras.models.model_from_json(f.read())
ae.load_weights(FLAGS.model_dir + "ae.h5")
print(ae.summary())  # Prints model summary

# -- Use AE for prediction
[enc, dec] = ae.predict(y.astype(np.float32))

# -- Data Extraction
n = 3200  # Number of patches retrieved

ys = []
encodings = []
with tf.Session() as sess:
    for _ in range(n // FLAGS.batch_size):
        ys.append(sess.run(x))
ys_ = []
for y in ys:
    [enc, _] = ae.predict(y.astype(np.float32))
    spatial_averaged = enc.mean(axis=(1, 2))
    encodings.extend(list(spatial_averaged))
    ys_.extend(y)
ys = ys_

encodings = np.array(encodings)


# -- Casper PCA run
# centered = encodings - encodings.mean(axis=0)
# cov = centered.transpose().dot(centered) / centered.shape[0]
# evals, evecs = np.linalg.eigh(cov)
# evals = np.flip(evals)
# evecs = np.flip(evecs, axis=1)

# TODO: Change for a TF based implementation
# PCA
# Min-max normalization
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import pandas as pd
import time, datetime

ts = time.time()

scaling_obj = RobustScaler()
scaled_encodings = scaling_obj.fit_transform(encodings)

pca_obj = PCA(n_components=None, random_state=333, svd_solver="auto")
out_pca = pd.DataFrame(pca_obj.fit_transform(scaled_encodings))
# print(out_pca.values)
df_pca = out_pca.values

# -- Send to TBoard
## Get working directory
PATH = os.getcwd()

## Path to save the embedding and checkpoints generated
LOG_DIR = (
    PATH
    + "/project-tensorboard/log-"
    + datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H:%M:%S")
    + "/"
)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print("Created directory: " + LOG_DIR)
    print("Run tensorboard as: tensorboard --logdir=" + LOG_DIR)

## TensorFlow Variable from data
tf_data = tf.Variable(df_pca)
## Running TensorFlow Session
with tf.Session() as sess:
    saver = tf.train.Saver([tf_data])
    sess.run(tf_data.initializer)
    saver.save(sess, os.path.join(LOG_DIR, "tf_data.ckpt"))
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = tf_data.name

    # TODO: Pass the lat,lon coordinates as labels
    # Link this tensor to its metadata(Labels) file
    # embedding.metadata_path = metadata

    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)


# ## Get working directory
# PATH = os.getcwd()
#
# ## Path to save the embedding and checkpoints generated
# LOG_DIR = PATH + '/project-tensorboard/log-1/'
# ## Load data
# df = pd.read_csv("scaled_data.csv",index_col =0)
# ## Load the metadata file. Metadata consists your labels. This is optional. Metadata helps us visualize(color) different clusters that form t-SNE
# metadata = os.path.join(LOG_DIR, 'df_labels.tsv')
# # Generating PCA and
# pca = PCA(n_components=50,
#          random_state = 123,
#          svd_solver = 'auto'
#          )
# df_pca = pd.DataFrame(pca.fit_transform(df))
# df_pca = df_pca.values
# ## TensorFlow Variable from data
# tf_data = tf.Variable(df_pca)
# ## Running TensorFlow Session
# with tf.Session() as sess:
#     saver = tf.train.Saver([tf_data])
#     sess.run(tf_data.initializer)
#     saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
#     config = projector.ProjectorConfig()
# # One can add multiple embeddings.
#     embedding = config.embeddings.add()
#     embedding.tensor_name = tf_data.name
#     # Link this tensor to its metadata(Labels) file
#     embedding.metadata_path = metadata
#     # Saves a config file that TensorBoard will read during startup.
#     projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
