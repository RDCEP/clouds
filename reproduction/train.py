import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import timeline
from tensorflow.profiler import ProfileOptionBuilder, Profiler
from pipeline import load as pipeline
import models
import subprocess
from os import path, mkdir


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
        "model_dir",
        help="/path/to/model/ to load and train or to save new model",
        default=None,
    )

    pipeline.add_pipeline_cli_arguments(p)

    p.add_argument(
        "--optimizer",
        help="type of optimizer to use for gradient descent. TODO (unused flag)",
        default="adam",
    )
    p.add_argument(
        "--epochs", type=int, help="Number of epochs to train for", default=10
    )
    p.add_argument(
        "--steps_per_epoch",
        metavar="N",
        help="Number of steps to train in each epoch ",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--summary_every",
        type=int,
        metavar="s",
        default=250,
        help="Number of steps per summary step",
    )

    p.add_argument(
        "--n_blocks",
        help="number of blocks in AE/disc. Each block ends with a strided convolution.",
        type=int,
        default=3,
    )
    p.add_argument(
        "--block_len",
        help="Number of layers in each block before strided convolution",
        type=int,
        default=1,
    )
    p.add_argument(
        "--base_dim",
        type=int,
        default=16,
        help="Depth of the first convolution, each block depth doubles",
    )
    p.add_argument("--batchnorm", action="store_true", default=False)

    p.add_argument(
        "--variational",
        default=False,
        action="store_true",
        help="Use a variational autoencoder.",
    )
    p.add_argument(
        "--vae_beta",
        default=1.0,
        type=float,
        help="Weight of KL-Divergence on VAE objective. Unused if not `variational`",
    )

    p.add_argument(
        "--discriminator",
        action="store_true",
        default=False,
        help=(
            "Augment autoencoder loss with the discriminator. Flag unused if "
            "continuing training in a directory with a saved discriminator as "
            "that will be loaded."
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
        "--perceptual",
        action="store_true",
        help="Pretrained classifier for perceptual loss",
    )
    p.add_argument(
        "--lambda_per", help="Weight of perceptual loss on AE objective", default=1.0
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

    commit = subprocess.check_output(["git", "describe", "--always"]).strip()
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Current Git Commit: {commit}")
    print("Flags:")
    for f in FLAGS.__dict__:
        print(f"\t{f}:{(25-len(f)) * ' '} {FLAGS.__dict__[f]}")
    print("\n", flush=True)

    if not path.isdir(FLAGS.model_dir):
        mkdir(FLAGS.model_dir)
        mkdir(path.join(FLAGS.model_dir, "timelines"))

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
        with tf.variable_scope("cmap"):
            # Whiten data with batchnorm so colors are more meaningful
            # t = tf.keras.layers.BatchNormalization()(t)
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


if __name__ == "__main__":
    FLAGS = get_flags()

    print("Building dataset...", flush=True)
    dataset = pipeline.load_data(
        FLAGS.data,
        FLAGS.shape,
        FLAGS.batch_size,
        FLAGS.read_threads,
        FLAGS.shuffle_buffer_size,
        FLAGS.prefetch,
    )
    shape = FLAGS.shape

    _, _, img = dataset.make_one_shot_iterator().get_next()

    # Colormap object maps our channels to normal rgb channels
    cmap = ColorMap(FLAGS.red_bands, FLAGS.green_bands, FLAGS.blue_bands)

    ### For each submodel (AE, Disc, Perceptual)
    # - Load or define it
    # - Define its losses and maybe add it to loss_ae
    # - Add the loss to tf summary
    # - Optimize it and add that to `train_ops`
    # - Add model to `save_models` so it will end up being saved.
    # Except for pretained models which do not need to be trained or saved

    print("building model and losses...", flush=True)
    with tf.name_scope("autoencoder"):
        ae = load_model(FLAGS.model_dir, "ae")
        if not ae:
            ae = models.autoencoder(
                shape=shape,
                base=FLAGS.base_dim,
                batchnorm=FLAGS.batchnorm,
                n_blocks=FLAGS.n_blocks,
                variational=FLAGS.variational,
                block_len=FLAGS.block_len,
            )

    if FLAGS.variational:
        latent_mean, latent_logvar, _, ae_img = ae(img)
        with tf.name_scope("kl_div"):
            kl_div = -0.5 * tf.reduce_sum(
                1 + latent_logvar - latent_mean ** 2 - tf.exp(latent_logvar)
            )
            tf.summary.scalar("kl_div", kl_div)
        loss_ae = mse = tf.reduce_mean(tf.square(img - ae_img))
        loss_ae += FLAGS.vae_beta * kl_div

    else:
        _, ae_img = ae(img)
        loss_ae = mse = tf.reduce_mean(tf.square(img - ae_img))

    cimg = cmap(img)
    camg = cmap(ae_img)
    tf.summary.image("original", cimg, FLAGS.display_imgs)
    tf.summary.image("autoencoded", camg, FLAGS.display_imgs)
    tf.summary.image("difference", cimg - camg, FLAGS.display_imgs)
    save_models = {"ae": ae}
    tf.summary.scalar("mse", mse)

    optimizer = tf.train.AdamOptimizer()  # TODO flag
    train_ops = []

    if FLAGS.discriminator:
        with tf.name_scope("discriminator"):
            disc = load_model(FLAGS.model_dir, "disc")
            if not disc:
                disc = models.discriminator(shape, FLAGS.n_blocks)
        with tf.name_scope("disc_loss"):
            di = disc(img)
            da = disc(ae_img)
            loss_disc = tf.reduce_mean(di - da)
            save_models["disc"] = disc
            tf.summary.scalar("loss_disc", loss_disc)

        loss_ae += FLAGS.lambda_disc * tf.reduce_mean(da)
        # Enforce Lipschitz discriminator scores between natural and autoencoded
        # manifolds by penalizing magnitude of gradient in this zone.
        # arxiv.org/pdf/1704.00028.pdf
        with tf.name_scope("gradient_penalty"):
            i = tf.random_uniform([1])
            between = img * i - ae_img * (1 - i)
            grad = tf.gradients(disc(between), between)[0]
            grad_norm = tf.reduce_mean(grad ** 2, axis=[1, 2, 3])
            penalty = tf.reduce_mean((grad_norm - 1) ** 2)
            tf.summary.scalar("gradient_penalty", penalty)

        loss_disc += penalty * FLAGS.lambda_gradient_penalty

        # Discriminator should be trained more to provide useful feedback
        train_disc = optimizer.minimize(loss_disc, var_list=disc.trainable_weights)
        train_ops += [train_disc] * FLAGS.n_critic

    if FLAGS.perceptual:
        # There is a minimum shape thats 139 or so but we only need early layers
        # Set input height / width to None so Keras doesn't complain
        inp_shape = None, None, 3
        per = our_models.classifier(inp_shape)
        with tf.name_scope("perceptual_loss"):
            pi = per(cmap(img))
            pa = per(cmap(ae_img))
            loss_per = tf.reduce_mean(tf.square(pi - pa))
            loss_ae += loss_per * FLAGS.lambda_per
            tf.summary.scalar("loss_per", loss_per)

    # Monitor AE gradients
    grads_and_vars = optimizer.compute_gradients(loss_ae, var_list=ae.trainable_weights)
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram("{}/grad_histogram".format(var.name), grad)
            tf.summary.scalar(
                "{}/grad/sparsity".format(var.name), tf.nn.zero_fraction(grad)
            )
    train_ops.append(optimizer.apply_gradients(grads_and_vars))
    tf.summary.scalar("loss_ae", loss_ae)

    # Save JSONs
    for m in save_models:
        with open(path.join(FLAGS.model_dir, f"{m}.json"), "w") as f:
            f.write(save_models[m].to_json())

    summary_op = tf.summary.merge_all()

    # Begin training
    print("Training...", flush=True)
    with tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=True
        )
    ) as sess:
        run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        profiler = Profiler(sess.graph)

        sess.run(
            tf.global_variables_initializer(),
            options=run_opts,
            run_metadata=run_metadata,
        )

        log_dir = path.join(FLAGS.model_dir, "summary")
        summary_writer = tf.summary.FileWriter(
            log_dir, graph=sess.graph if not path.exists(log_dir) else None
        )

        for e in range(FLAGS.epochs):
            print("Starting epoch %d" % e, flush=True)

            for s in range(FLAGS.steps_per_epoch):
                sess.run(train_ops, options=run_opts, run_metadata=run_metadata)

                if s % FLAGS.summary_every == 0:
                    total_step = e * FLAGS.steps_per_epoch + s
                    summary_writer.add_summary(sess.run(summary_op), total_step)
                    profiler.add_step(total_step, run_metadata)
                    profiler.profile_graph(
                        options=(
                            ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory())
                            .with_step(total_step)
                            .with_timeline_output(
                                path.join(FLAGS.model_dir, "timelines", "t.json")
                            )
                            .build()
                        )
                    )
                    profiler.advise(
                        {
                            "ExpensiveOperationChecker": {},
                            "AcceleratorUtilizationChecker": {},
                            "OperationChecker": {},
                        }
                    )

            for m in save_models:
                save_models[m].save_weights(path.join(FLAGS.model_dir, f"{m}.h5"))
