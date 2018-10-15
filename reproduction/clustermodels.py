from argparse import ArgumentParser
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.externals import joblib
from pipeline import load
from tqdm import trange

import tensorflow as tf
import os


p = ArgumentParser()
p.add_argument("out_dir")
p.add_argument("encoder")
p.add_argument("--max_steps", type=int, default=100000)
p.add_argument("--n_clusters", type=int, default=60)
p.add_argument("--summary_every", type=int, default=10)
p.add_argument("--save_every", type=int, default=50)
load.add_pipeline_cli_arguments(p)

FLAGS = p.parse_args()
os.makedirs(FLAGS.out_dir, exist_ok=True)


with open(os.path.join(FLAGS.encoder, "encoder.json"), "r") as f:
    encoder = tf.keras.models.model_from_json(f.read())

ds = load.load_data(
    FLAGS.data,
    FLAGS.shape,
    FLAGS.batch_size,
    FLAGS.read_threads,
    FLAGS.shuffle_buffer_size,
    FLAGS.prefetch,
    not FLAGS.no_augment_flip,
    not FLAGS.no_augment_rotate,
)

_, _, imgs = ds.make_one_shot_iterator().get_next()
codes = encoder(imgs)

models = {
    "sparse_dict": MiniBatchDictionaryLearning(FLAGS.n_clusters),
    "kmeans": MiniBatchKMeans(FLAGS.n_clusters)
}

save_dir = os.listdir(FLAGS.out_dir)
step = 0

for m in models:
    saved = sorted([saved for saved in save_dir if m in saved])
    if saved:
        s = os.path.join(FLAGS.out_dir, saved[-1])
        print("Loading", m, "from", s)
        models[m] = joblib.load(s)
        step = int(s.split("-")[1].split(".joblib")[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    encoder.load_weights(os.path.join(FLAGS.encoder, "encoder.h5"))

    for step in trange(step, FLAGS.max_steps):
        c = sess.run(codes)
        c = c.reshape((FLAGS.batch_size, -1))

        for m in models:
            models[m].partial_fit(c)

        if step % FLAGS.summary_every == 0:
            pass

        if step % FLAGS.save_every == 0:
            seen = step * FLAGS.batch_size
            for m in models:
                path = os.path.join(FLAGS.out_dir, "%s-%07d.joblib" % (m, seen))
                joblib.dump(models[m], path)
