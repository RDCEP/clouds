import tensorflow as tf
from argparse import ArgumentParser


p = ArgumentParser()
p.add_argument("data_glob")

FLAGS = p.parse_args()

img_width, img_height, n_bands = 64, 64, 7

features = {
    f"b{i+1}": tf.FixedLenFeature((img_width, img_height), tf.float32)
    for i in range(n_bands)
}


def stack_bands(x):
    return tf.stack([x[f"b{i+1}"] for i in range(n_bands)])

batch_size = 32

data = (
    tf.data.Dataset.list_files(FLAGS.data_glob)
    #.apply(tf.contrib.data.shuffle_ad_repeat(500))
    .flat_map(tf.data.TFRecordDataset)
    .map(lambda serialized: tf.parse_single_example(serialized, features))
    .map(stack_bands)
    .batch(batch_size)
)

x = data.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    print(sess.run(x))
