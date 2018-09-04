import tensorflow as tf
from os import path
from argparse import ArgumentParser
from google.protobuf import text_format
from IPython import embed

p = ArgumentParser()
p.add_argument("model_dir", help="Directory of model to convert to ckpt / graph def")
FLAGS = p.parse_args()

# tf.keras.backend.set_learning_phase(0)

with open(path.join(FLAGS.model_dir, "ae.json")) as f:
    m = tf.keras.models.model_from_json(f.read())
m.load_weights(path.join(FLAGS.model_dir, "ae.h5"))

saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
saver.save(sess, path.join(FLAGS.model_dir, "ckpt"))

tf.train.write_graph(sess.graph_def, FLAGS.model_dir, "model.GraphDef")

with open(path.join(FLAGS.model_dir, "model.GraphDef"), "r") as f:
    gd = tf.GraphDef()
    text_format.Merge(f.read(), gd)
