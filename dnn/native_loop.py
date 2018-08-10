import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import timeline
import pipeline
import model
import subprocess
from os import path, mkdir


def get_flags():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-d", "--data" help="pattern to pick up tf records files", required=True)
    p.add_argument("-b", "--batch_size", type=int, help=" ", default=32)
    p.add_argument(
        "model_dir",
        help="/path/to/model/ to load and train or to save new model",
        default=None,
    )
    p.add_argument(
        "-o",
        "--optimizer",
        help="type of optimizer to use for gradient descent",
        default="adam",
    )
    p.add_argument("-l", "--loss", help=" ", default="mean_squared_error")
    p.add_argument(
        "-spe",
        "--steps_per_epoch",
        help="Number of steps to train in each epoch ",
        type=int,
        default=1000,
    )
    p.add_argument(
        "-e", "--epochs", type=int, help="Number of epochs to train for", default=10
    )
    p.add_argument("-nm", "--new_model", help="Name of model in model.py to use")
    p.add_argument(
        "-sh",
        "--shape",
        nargs=3,
        type=int,
        help="Shape of input image",
        default=(64, 64, 7),
    )
    p.add_argument(
        "--discriminator",
        action="store_true",
        default=False,
        help="Augment autoencoder loss with a discriminator",
    )
    p.add_argument("--perceptual", help="Pretrained")

    FLAGS = p.parse_args()
    if FLAGS.model_dir[-1] != "/":
        FLAGS.model_dir += "/"

    commit = subprocess.check_output(["git", "describe", "--always"]).strip()
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Current Git Commit: {commit}")
    print("Flags:")
    for f in FLAGS.__dict__:
        print(f"\t{f}:{(20-len(f)) * ' '} {FLAGS.__dict__[f]}")

    return FLAGS


def load_data(data_glob, shape, batch_size):

    img_width, img_height, n_bands = shape

    features = {
        f"b{i+1}": tf.FixedLenFeature((img_width, img_height), tf.float32)
        for i in range(n_bands)
    }

    def stack_bands(x):
        return tf.stack([x[f"b{i+1}"] for i in range(n_bands)], axis=2)

    return (
        tf.data.Dataset.list_files(FLAGS.data_glob)
        .flat_map(tf.data.TFRecordDataset)
        .map(lambda serialized: tf.parse_single_example(serialized, features))
        .map(stack_bands)
        .batch(FLAGS.batch_size)
    )


def load_or_define_ae(model_dir, new_model, shape):
    """Loads existing model or defines a new one based on flags. Models are
    retrieved from `model.py`
    """
    if not path.isdir(model_dir):
        mkdir(model_dir)

    model_file = path.join(model_dir, "model.h5")

    if path.exists(model_file):
        print(f"Loading existing model")
        ae = keras.models.load_model(model_file)
    else:
        if new_model:
            build_model = getattr(model, new_model)
        else:
            build_model = model.autoencoder

        print(f"Defining new model from {build_model}")
        _, ae = build_model(shape)


def load_or_define_disc():
    raise NotImplementedError()


def load_or_define_per():
    raise NotImplementedError()


if __name__ == "__main__":
    FLAGS = get_flags()
    data = load_data(FLAGS.data_glob, FLAGS.shape, FLAGS.batch_size)

    ae = load_or_define_ae(FLAGS.model_dir, FLAGS.new_model, FLAGS.shape)

    optimizer = tf.train.AdamOptimizer()  # TODO configurable
    train_ops = []

    img = data.make_one_shot_iterator().get_next()
    ae_img = ae(img)
    loss_ae = mse = tf.reduce_mean(tf.square(img - ae_img))

    summary_imgs = {"img": tf.variable(img), "ae_img": tf.variable(ae_img)}
    summary_nums = {"mse": mse}

    if FLAGS.discriminator:
        disc = load_or_define_disc(FLAGS.model_dir, FLAGS.shape)

        di = disc(img)
        da = disc(ae_img)

        loss_disc = di - da
        loss_ae += -da

        summary_nums["loss_disc"] = loss_disc
        # TODO discriminator should be trained 5 times as much?
        train_ops.append(optimizer.minimize(loss_disc, var_list=disc.trainable_weights))

    if FLAGS.perceptual:
        per = load_or_define_per(FLAGS.model_dir, FLAGS.shape)

        pi = per(img)
        pa = per(ae_img)
        loss_per = tf.reduce_mean(tf.square(pi - pa))
        loss_ae += loss_per

        summary_nums["loss_per"] = loss_per

    summary_nums["loss_ae"] = loss_ae
    train_ops.append(optimizer.minimize(loss_ae, var_list=ae.trainable_weights))

    for m in save_models:
        with open(path.join(FLAGS.model_dir, f"{m}.json"), f):
            f.write(save_models[m].to_json())

    with tf.Session() as sess:

        tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            path.join(FLAGS.model_dir, "summary"), sess.graph
        )

        for _ in range(FLAGS.epochs):

            for _ in range(FLAGS.steps_per_epoch):
                sess.run(train_ops)

            for m in save_models:
                save_models[m].save_weights(path.join(FLAGS.model_dir, f"{m}.h5"))


#
