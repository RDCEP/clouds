import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import timeline
import pipeline
import model as our_models
import subprocess
from tensorflow.contrib.data import shuffle_and_repeat, batch_and_drop_remainder
from os import path, mkdir
import os


def get_flags():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "This program trains an autoencoder on satellite image data of clouds. "
            "It uses a convolutional autoencoder trying ot minimize "
            "mean-squared-error plus configurable loss from a discriminator or "
            "perceptual difference given a pretrained network."
        ),
    )
    p.add_argument(
        "--data", help="pattern to pick up tf records files", required=True, nargs="+"
    )
    p.add_argument(
        "--hdf_fields",
        nargs="+",
        help=(
            "fields to select from hdf tf record. If no fields are provided "
            "then tf records are parsed with field names b0...bK where K is the"
            "number of dimensions indicated by `shape` I.e. tiff data. For hdf "
            "data, the number of channels will be determined consulting meta_json"
        ),
    )
    p.add_argument(
        "--meta_json",
        help="json file giving hdf meta-data describing dimensionality of hdf data.",
    )
    p.add_argument("--batch_size", type=int, help=" ", default=32)
    p.add_argument(
        "model_dir",
        help="/path/to/model/ to load and train or to save new model",
        default=None,
    )
    p.add_argument(
        "--optimizer",
        help="type of optimizer to use for gradient descent. TODO (unused flag)",
        default="adam",
    )
    p.add_argument(
        "--steps_per_epoch",
        metavar="N",
        help="Number of steps to train in each epoch ",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--n_layers",
        help="number of strided convolution layers in AE / disc",
        type=int,
        default=3,
    )
    p.add_argument(
        "--epochs", type=int, help="Number of epochs to train for", default=10
    )
    p.add_argument(
        "--new_model",
        default="",
        help=(
            "Name of model in model.py to use for the autoencoder. Flag "
            "unused if the training in a directory with a saved autoencoder."
        ),
    )
    p.add_argument(
        "--shape", nargs=3, type=int, help="Shape of input image", default=(64, 64, 7)
    )
    p.add_argument(
        "--discriminator",
        default="",
        help=(
            "Augment autoencoder loss with the discriminator with this name"
            "defaults to models.discriminator(...) if name is invalid. "
            "Flag unused if continuing training in a directory with a saved "
            "discriminator."
        ),
    )
    p.add_argument(
        "--lambda_disc",
        type=float,
        default=0.01,
        help="Weight of discriminative loss on AE objective",
    )
    p.add_argument(
        "--lambda_gradient_penalty",
        type=float,
        default=10,
        help="Weight of 1-lipschitz constraint on discriminator objective",
    )
    p.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="number of discriminator updates per autoencoder update",
    )
    p.add_argument(
        "--lambda_per", help="Weight of perceptual loss on AE objective", default=1.0
    )
    p.add_argument(
        "--perceptual",
        action="store_true",
        help="Pretrained classifier for perceptual loss",
    )
    p.add_argument(
        "--display_imgs",
        metavar="N",
        type=int,
        default=8,
        help="Number of images to display on tensorboard",
    )
    p.add_argument(
        "--red_bands",
        type=int,
        metavar="r",
        nargs="+",
        default=[1, 4, 5, 6],
        help=(
            "0-indexed bands to map to red for tensorboard display and for input"
            " to pretrained classifiers"
        ),
    )
    p.add_argument(
        "--green_bands",
        type=int,
        metavar="g",
        nargs="+",
        default=[0],
        help=(
            "0-indexed bands to map to green for tensorboard display and for "
            "input to pretrained classifiers"
        ),
    )
    p.add_argument(
        "--blue_bands",
        type=int,
        metavar="b",
        nargs="+",
        default=[2, 3],
        help=(
            "0-indexed bands to map to red for tensorboard display and for "
            "input to pretrained classifiers"
        ),
    )

    FLAGS = p.parse_args()
    if FLAGS.model_dir[-1] != "/":
        FLAGS.model_dir += "/"

    if bool(FLAGS.hdf_fields) != bool(FLAGS.meta_json):
        raise ValueError("`--meta_json` must be used with `--hdf_fields`")

    commit = subprocess.check_output(["git", "describe", "--always"]).strip()
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Current Git Commit: {commit}")
    print("Flags:")
    for f in FLAGS.__dict__:
        print(f"\t{f}:{(25-len(f)) * ' '} {FLAGS.__dict__[f]}")
    print("\n")

    if not path.isdir(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)

    return FLAGS


class ColorMap:
    def __init__(self, green_bands, red_bands, blue_bands):
        self.greens = green_bands
        self.reds = red_bands
        self.blues = blue_bands

    def __call__(self, t):
        """Converts the input tensor channels to RGB channels
        [batch, height, width, bands] -> [batch, height, width, rgb]
        """
        # TODO Consider adding batchnorm here to whiten output bands, because
        # reflectances are in [-100, 16000] while tf rgb is probably [0, 1]
        r = tf.reduce_mean(select_channels(t, self.reds), axis=3)
        g = tf.reduce_mean(select_channels(t, self.greens), axis=3)
        b = tf.reduce_mean(select_channels(t, self.blues), axis=3)
        return tf.stack([r, g, b], axis=3)


def select_channels(t, chans):
    """Select `chans` channels from the last dimension of a tensor.
    Equivalent to array[:,:,..,chans]
    """
    t = tf.transpose(t)
    t = tf.gather(t, chans)
    t = tf.transpose(t)
    return t


def load_tif_data(data_files, shape, batch_size):
    return (
        tf.data.TFRecordDataset(data_files)
        .apply(shuffle_and_repeat(500))
        .map(pipeline.parse_tfr_fn(shape))
        .apply(batch_and_drop_remainder(batch_size))
    )


def load_hdf_data(data_files, shape, batch_size, hdf_fields, meta_json):

    chans, parser = pipeline.hdf_tfr_fn(hdf_fields, meta_json)

    return chans, (
        tf.data.TFRecordDataset(data_files)
        .apply(shuffle_and_repeat(500))
        .map(parser)
        .interleave(pipeline.patchify_fn(shape[0], shape[1], chans), cycle_length=4)
        .shuffle(10000)
        .apply(batch_and_drop_remainder(batch_size))
    )


def load_model(model_dir, name):
    json = path.join(model_dir, name + ".json")
    weights = path.join(model_dir, name + ".h5")

    if path.exists(json) and path.exists(weights):
        with open(json, "r") as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights(weights)
        print(f"model loaded from {model_dir} {name}")
        return model

    return None


def define_model(new_model, default, shape, n_layers):
    builder = getattr(our_models, new_model, getattr(our_models, default))
    return builder(shape, n_layers)


if __name__ == "__main__":
    FLAGS = get_flags()

    if FLAGS.hdf_fields:
        chans, dataset = load_hdf_data(
            FLAGS.data, FLAGS.shape, FLAGS.batch_size, FLAGS.hdf_fields, FLAGS.meta_json
        )
        if FLAGS.shape[-1] != chans:
            print(
                "WARNING: provided channels do not match hdf choices. "
                "Correcting to %d channels..." % chans
            )
        FLAGS.shape = (*FLAGS.shape[:2], chans)

    else:
        dataset = load_tif_data(FLAGS.data, FLAGS.shape, FLAGS.batch_size)

    img = dataset.make_one_shot_iterator().get_next()

    # Colormap object maps our channels to normal rgb channels
    cmap = ColorMap(FLAGS.red_bands, FLAGS.green_bands, FLAGS.blue_bands)

    ### For each submodel (AE, Disc, Perceptual)
    # - Load or define it
    # - Define its losses and maybe add it to loss_ae
    # - Add the loss to tf summary
    # - Optimize it and add that to `train_ops`
    # - Add model to `save_models` so it will end up being saved.
    # Except for pretained models which do not need to be trained or saved

    with tf.name_scope("autoencoder"):
        ae = load_model(FLAGS.model_dir, "ae")
        if not ae:
            ae = define_model(
                FLAGS.new_model, "autoencoder", FLAGS.shape, FLAGS.n_layers
            )

    _, ae_img = ae(img)

    loss_ae = mse = tf.reduce_mean(tf.square(img - ae_img))

    tf.summary.image("original", cmap(img), FLAGS.display_imgs)
    tf.summary.image("autoencoded", cmap(ae_img), FLAGS.display_imgs)
    tf.summary.image("difference", cmap(img) - cmap(ae_img), FLAGS.display_imgs)
    save_models = {"ae": ae}
    tf.summary.scalar("mse", mse)

    optimizer = tf.train.AdamOptimizer()  # TODO flag
    train_ops = []


    if FLAGS.discriminator:
        with tf.name_scope("discriminator"):
            disc = load_model(FLAGS.model_dir, "disc")
            if not disc:
                disc = define_model(
                    FLAGS.discriminator, "discriminator", FLAGS.shape, FLAGS.n_layers
                )

        di = disc(img)
        da = disc(ae_img)

        loss_disc = tf.reduce_mean(di - da)
        loss_ae += FLAGS.lambda_disc * tf.reduce_mean(da)

        save_models["disc"] = disc
        tf.summary.scalar("loss_disc", loss_disc)
        # Enforce Lipschitz discriminator scores between natural and autoencoded
        # manifolds by penalizing magnitude of gradient in this zone.
        # arxiv.org/pdf/1704.00028.pdf
        i = tf.random_uniform([1])
        between = img * i - ae_img * (1 - i)
        grad = tf.gradients(disc(between), between)[0]
        grad_norm = tf.reduce_mean(grad ** 2, axis=[1, 2, 3])
        penalty = tf.reduce_mean((grad_norm - 1) ** 2)
        loss_disc += penalty * FLAGS.lambda_gradient_penalty
        tf.summary.scalar("gradient_penalty", penalty)

        # Discriminator should be trained more to provide useful feedback
        train_disc = optimizer.minimize(loss_disc, var_list=disc.trainable_weights)
        train_ops += [train_disc] * FLAGS.n_critic


    if FLAGS.perceptual:
        # There is a minimum shape thats 139 or so but we only need early layers
        # Set input height / width to None so Keras doesn't complain
        inp_shape = None, None, 3
        per = our_models.classifier(inp_shape)
        pi = per(cmap(img))
        pa = per(cmap(ae_img))
        loss_per = tf.reduce_mean(tf.square(pi - pa))
        loss_ae += loss_per * FLAGS.lambda_per
        tf.summary.scalar("loss_per", loss_per)


    tf.summary.scalar("loss_ae", loss_ae)
    train_ops.append(optimizer.minimize(loss_ae, var_list=ae.trainable_weights))

    # Save JSONs
    for m in save_models:
        with open(path.join(FLAGS.model_dir, f"{m}.json"), "w") as f:
            f.write(save_models[m].to_json())

    summary_op = tf.summary.merge_all()

    # Begin training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        log_dir = path.join(FLAGS.model_dir, "summary")
        summary_writer = tf.summary.FileWriter(
            log_dir, graph=sess.graph if not path.exists(log_dir) else None
        )

        for e in range(FLAGS.epochs):
            print("Starting epoch %d" % e)

            for s in range(FLAGS.steps_per_epoch):
                sess.run(train_ops)

                if s % 50 == 0:
                    summary_writer.add_summary(sess.run(summary_op))

            for m in save_models:
                save_models[m].save_weights(path.join(FLAGS.model_dir, f"{m}.h5"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(ae_img))
