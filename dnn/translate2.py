import tensorflow as tf
from google.protobuf import text_format

gd_file = "out/m4/graph_def.pb"
ck_file = "out/m4/model.ckpt2"


with tf.Graph().as_default():
    with tf.gfile.FastGFile(gd_file, "r") as f:
        graph_def = tf.GraphDef()
        text_format.Merge(f.read(), graph_def)

        tf.import_graph_def(graph_def, name="")
        saver = tf.train.Saver()
