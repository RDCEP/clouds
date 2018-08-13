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
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-d", "--data", help="pattern to pick up tf records files", required=True, nargs="+",
    )
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
    p.add_argument(
        "-nm", "--new_model", help="Name of model in model.py to use", default=""
    )
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
        default="",
        help="Augment autoencoder loss with a discriminator",
    )
    p.add_argument("-ld", "--lambda_disc", type=float, default=0.01)
    p.add_argument("-lgp", "--lambda_gradient_penalty", type=float, default=10)
    p.add_argument(
        "-nc",
        "--n_critic",
        type=int,
        default=5,
        help="number of discriminator updates per autoencoder update",
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
        print(f"\t{f}:{(25-len(f)) * ' '} {FLAGS.__dict__[f]}")
    print("\n")

    if not path.isdir(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)

    return FLAGS


def load_data(data_files, shape, batch_size):
    img_width, img_height, n_bands = shape
    return (
        tf.data.TFRecordDataset(data_files)
        .apply(shuffle_and_repeat(500))
        .map(pipeline.parse_tfr_fn(shape))
        .apply(batch_and_drop_remainder(batch_size))
    )


def load_model(model_dir, name):
    json = path.join(model_dir, name + ".json")
    weights = path.join(model_dir, name + ".h5")

    if path.exists(json) and path.exists(weights):
        with open(json, "r") as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights(weights)
        return model

    return None


def define_model(new_model, default, shape):
    builder = getattr(our_models, new_model, getattr(our_models, default))
    return builder(shape)


if __name__ == "__main__":
    FLAGS = get_flags()
    img = (
        load_data(FLAGS.data, FLAGS.shape, FLAGS.batch_size)
        .make_one_shot_iterator()
        .get_next()
    )

    dataset = load_data(FLAGS.data, FLAGS.shape, FLAGS.batch_size)

    with tf.name_scope("autoencoder"):
        ae = load_model(FLAGS.model_dir, "ae")
        if not ae:
            ae = define_model(FLAGS.new_model, "autoencoder", FLAGS.shape)

    _, ae_img = ae(img)

    tf.summary.image(
        "img", tf.expand_dims(tf.reduce_mean(img, axis=3), axis=-1), max_outputs=5
    )
    tf.summary.image(
        "ae_img", tf.expand_dims(tf.reduce_mean(ae_img, axis=3), axis=-1), max_outputs=5
    )
    tf.summary.image(
        "difference",
        tf.expand_dims(tf.reduce_mean(ae_img - img, axis=3), axis=-1),
        max_outputs=5
    )

    loss_ae = mse = tf.reduce_mean(tf.square(img - ae_img))

    save_models = {"ae": ae}
    # summary_imgs = {"img": tf.Variable(img), "ae_img": tf.Variable(ae_img)}
    tf.summary.scalar("mse", mse)
    optimizer = tf.train.AdamOptimizer()  # TODO flag
    train_ops = []

    if FLAGS.discriminator:
        with tf.name_scope("discriminator"):
            disc = load_model(FLAGS.model_dir, "disc")
            if not disc:
                disc = define_model(FLAGS.discriminator, "discriminator", FLAGS.shape)

        di = disc(img)
        da = disc(ae_img)

        loss_disc = tf.reduce_mean(di - da)
        loss_ae += FLAGS.lambda_disc * tf.reduce_mean(da)

        save_models["disc"] = disc
        tf.summary.scalar("loss_disc", loss_disc)
        # Enforce Lipschitz discriminator scores between natural and autoencoded manifolds
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
        per = define_model(FLAGS.perceptual, "perceptual", FLAGS.shape)

        pi = per(img)
        pa = per(ae_img)
        loss_per = tf.reduce_mean(tf.square(pi - pa))
        loss_ae += loss_per

        tf.summary.scalar("loss_per", loss_per)

    tf.summary.scalar("loss_ae", loss_ae)
    train_ops.append(optimizer.minimize(loss_ae, var_list=ae.trainable_weights))

    # Save JSONs
    for m in save_models:
        with open(path.join(FLAGS.model_dir, f"{m}.json"), "w") as f:
            f.write(save_models[m].to_json())

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        log_dir = path.join(FLAGS.model_dir, "summary")
        summary_writer = tf.summary.FileWriter(
            log_dir, graph=sess.graph if path.exists(log_dir) else None
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
