import tensorflow as tf
from os import path
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("model_dir", help="Directory of model to convert to ckpt / graph def")
FLAGS = p.parse_args()

m = tf.keras.models.load_model(path.join(FLAGS.model_dir, "model.h5"))
saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
saver.save(sess, FLAGS.model_dir)

tf.train.write_graph(
    sess.graph_def,
    FLAGS.model_dir,
    "model.GraphDef",
    as_text=True
)
