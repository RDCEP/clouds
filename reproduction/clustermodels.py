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
p.add_argument("model", choices=["sparse_dict", "kmeans"])
p.add_argument("--latent", choices=["flatten", "spatial_mean"], default="spatial_mean")
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

# Get the Latent vectors to cluster
_, _, imgs = ds.make_one_shot_iterator().get_next()
# codes = encoder(imgs)

# if FLAGS.latent == "spatial_mean":
#    codes = tf.reduce_mean(codes, axis=(1, 2))

# elif FLAGS.latent == "flatten":
#   codes = tf.reshape(codes, [FLAGS.batch_size, -1])

# else:
#   raise ValueError("Invalid latent vector treatment", FLAGS.latent)


# Define clustering model or load it if already saved.
def get_latest(model_dir, model):
    saved = sorted([saved for saved in model_dir if model in saved])
    if saved:
        s = os.path.join(model_dir, saved[-1])
        step = int(s.split("-")[1].split(".joblib")[0])
        return step, joblib.load(s)

    if model == "sparse_dict":
        model = MiniBatchDictionaryLearning(FLAGS.n_clusters)

    elif model == "kmeans":
        model = MiniBatchKMeans(FLAGS.n_clusters)

    else:
        raise ValueError("Invalid model", FLAGS.model)

    return 0, model

step, model = get_latest(FLAGS.out_dir, FLAGS.model)

# Start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    encoder.load_weights(os.path.join(FLAGS.encoder, "encoder-"+FLAGS.encoder_step+".h5"))

    for step in range(step, FLAGS.max_steps):
        # c = sess.run(codes)
        i = sess.run(imgs)
        c = encoder.predict(i).mean(axis=(1, 2))

        model.partial_fit(c)

        if step % FLAGS.summary_every == 0:
            print("Step", step, flush=True)

        if step % FLAGS.save_every == 0:
            seen = step * FLAGS.batch_size
            path = os.path.join(FLAGS.out_dir, "%s-%07d.joblib" % (FLAGS.model, seen))
            joblib.dump(model, path)
