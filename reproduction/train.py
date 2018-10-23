import argparse
import tensorflow as tf
import tensorflow.keras as keras
from horovod import tensorflow as hvd
from tensorflow.python.client import timeline
from tensorflow.profiler import ProfileOptionBuilder, Profiler
from tensorflow.image import image_gradients
from pipeline import load as pipeline
import models
import subprocess
from os import path, makedirs, listdir


def get_flags(verbose):
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
    p.add_argument("--log_device_placement", action="store_true", default=False)

    pipeline.add_pipeline_cli_arguments(p)

    p.add_argument(
        "--channel_order",
        choices=["channels_last", "channels_first"],
        default="channels_last",
    )

    p.add_argument(
        "--gaussian_noise",
        type=float,
        metavar="stdev",
        default=0,
        help="stdev of gaussian noise added before AE",
    )
    p.add_argument(
        "--salt_pepper",
        type=float,
        metavar="pct",
        default=0,
        help="percentage of pixels hit with salt and pepper noise before AE",
    )

    p.add_argument("--max_steps", metavar="steps", type=int, default=1000000)
    p.add_argument("--save_every", metavar="steps", type=int, default=1000)
    p.add_argument("--summary_every", metavar="steps", type=int, default=250)

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
    p.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Stride length and factor which depth increases when changing of scales.",
    )
    p.add_argument("--batchnorm", action="store_true", default=False)
    p.add_argument(
        "--dense_ae", type=int, help="Dimension of dense bottleneck (Fixes patch size)."
    )
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
        "--image_loss_weights",
        nargs=4,
        type=float,
        default=(1, 0, 0, 0),
        metavar=("mse", "mae", "hfe", "mssim"),
        help=(
            "Weights for image losses: Mean squared error, Mean absolute error, Mean "
            "High Frequency Error (HFE), and Multi-Scale Structural Similarity (MSSIM). "
            "HFE is the mean absolute error of the x and y gradients of the image. It "
            "emphasizes edges. MS-SIM is the geometric average of the similarity of "
            "means, similarity of standard deviations, and correlation."
        ),
    )
    p.add_argument(
        "--autoencoder_adam",
        type=float,
        nargs=3,
        metavar=("lr", "b1", "b2"),
        default=(0.0001, 0, 0.9),
        help="Adam optimizer learning rate, beta1, beta2 for autoencoder",
    )
    p.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help=(
            "Adversarial Autencoder, Decoder is also a GAN and should be able to "
            "generate convincing images from gaussian noise."
        ),
    )
    p.add_argument(
        "--discriminator_adam",
        type=float,
        nargs=3,
        metavar=("lr", "b1", "b2"),
        default=(0.0004, 0, 0.9),
        help="Adam optimizer learning rate, beta1, beta2 for discriminator",
    )
    p.add_argument(
        "--lambda_disc",
        metavar="l",
        type=float,
        default=0.001,
        help="Weight of discriminative loss on AE objective",
    )
    p.add_argument(
        "--lambda_gradient_penalty",
        metavar="l",
        type=float,
        default=10,
        help="Weight of 1-lipschitz constraint on discriminator objective",
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
    p.add_argument(
        "--no_grad_histogram",
        action="store_true",
        help="Display gradient histogram on tensorboard",
    )

    FLAGS = p.parse_args()
    if FLAGS.model_dir[-1] != "/":
        FLAGS.model_dir += "/"

    if verbose:
        commit = subprocess.check_output(["git", "describe", "--always"]).strip()
        print("Tensorflow version:", tf.__version__)
        print("Current Git Commit:", commit)
        print("Flags:")
        for f in FLAGS.__dict__:
            print("\t", f, " " * (25 - len(f)), FLAGS.__dict__[f])
        print("\n", flush=True)

    makedirs(path.join(FLAGS.model_dir, "timelines"), exist_ok=True)

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


def add_noise(imgs, salt_pepper, gaussian_noise):
    """Noise input image for denoising autoencoders.
    Args:
        salt_pepper: Percentage of pixels to be salt or pepper noise. Half of these pixels
            are set to batch maximum, and half to batch minimum.
        gaussian_noise: Stdev of normal noise added to every channel and pixel.
    """
    if gaussian_noise:
        imgs += tf.random_normal(imgs.shape, stddev=gaussian_noise)

    if salt_pepper:
        # Spatial maximum / minimum for each element in batch and for every channel
        mx = tf.expand_dims(tf.expand_dims(tf.reduce_max(imgs, axis=(1, 2)), 1), 1)
        mn = tf.expand_dims(tf.expand_dims(tf.reduce_min(imgs, axis=(1, 2)), 1), 1)
        # Select random pixels
        noised = tf.cast(tf.random_uniform(imgs.shape[:3]) < salt_pepper, tf.float32)
        noised = tf.expand_dims(noised, 3)
        salt = tf.cast(tf.random_uniform(imgs.shape[:3]) > 0.5, tf.float32)
        salt = tf.expand_dims(salt, 3)
        imgs *= 1 - noised  # Zero out noised pixels
        # Apply salt and pepper
        imgs += noised * mx * salt
        imgs += noised * mn * (1 - salt)

    return imgs


def load_model_def(model_dir, name):
    json = path.join(model_dir, name + ".json")
    if path.exists(json):
        with open(json, "r") as f:
            model = tf.keras.models.model_from_json(f.read())
        print("model definition loaded from", json)
        return model
    return None


def load_latest_model_weights(model, model_dir, name):
    saved = [m for m in listdir(model_dir) if ".h5" in m and name in m]
    if saved:
        s = sorted(saved)[-1]
        model.load_weights(path.join(model_dir, s))
        step = int(s.split("-")[1].replace(".h5", ""))
        return step
    else:
        raise ValueError("No weights for ", model, "in", model_dir)


def loss_fn(name, weight, fn, **kwargs):
    """Helper fn to add summary, scope, and nonzero weight condition.
    """
    if weight:
        with tf.name_scope(name):
            loss = fn(**kwargs)
            tf.summary.scalar(name, loss)
            return loss * weight
    return 0


def new_adam_optimizer(size, lr, b1, b2):
    opt = tf.train.AdamOptimizer(lr * size, lr, b1, b2)
    return hvd.DistributedOptimizer(opt)


def image_losses(img, ae_img, w_mse, w_mae, w_hfe, w_ssim):
    """Applies the 4 image losses.
    - Mean Square Error (L2 Loss)
    - Mean Absolute Error
    - High Frequency Error (mean abs difference after passing edge detectors)
    - Multi-scale structural similarity (ensure similar image mean/stdev/correlation)
    """

    def hfe():
        dx_img, dy_img = image_gradients(img)
        dx_ae_img, dy_ae_img = image_gradients(ae_img)
        return tf.reduce_mean(tf.abs(dx_img - dx_ae_img) + tf.abs(dy_img - dy_ae_img))

    def msssim():
        # BUG why does it work on only these power factors?
        s = tf.image.ssim_multiscale(img, ae_img, max_val=5, power_factors=[1, 1, 1])
        return 1 - tf.reduce_mean(s)

    mse = lambda: tf.reduce_mean((img - ae_img) ** 2)
    mae = lambda: tf.reduce_mean(tf.abs(img - ae_img))

    l = 0
    with tf.name_scope("image_loss"):
        l += loss_fn("mean_square_error", w_mse, mse)
        l += loss_fn("mean_abs_error", w_mae, mae)
        l += loss_fn("high_frequency_error", w_hfe, hfe)
        l += loss_fn("ms_ssim", w_ssim, msssim)
    return l


if __name__ == "__main__":
    hvd.init()
    FLAGS = get_flags(hvd.rank() == 0)
    global_step = tf.train.get_or_create_global_step()

    print("Building dataset...", flush=True)
    dataset = pipeline.load_data(
        FLAGS.data,
        FLAGS.shape,
        FLAGS.batch_size,
        FLAGS.read_threads,
        FLAGS.shuffle_buffer_size,
        FLAGS.prefetch,
        not FLAGS.no_augment_flip,
        not FLAGS.no_augment_rotate,
        distribute=(hvd.size(), hvd.rank()),
    )
    shape = FLAGS.shape

    _, _, img = dataset.make_one_shot_iterator().get_next()

    if FLAGS.channel_order == "channels_first":
        img = tf.transpose(img, perm=[0, 3, 1, 2])
        shape = shape[2], * shape[:2]
        print("sh", shape)

    # DEBUG: I have no idea if this helps (remove if unneeded)
    tf.keras.backend.set_image_data_format(FLAGS.channel_order)

    # Colormap object maps our channels to normal rgb channels
    cmap = ColorMap(FLAGS.red_bands, FLAGS.green_bands, FLAGS.blue_bands)

    ### For each submodel (AE, Disc, Perceptual)
    # - Load or define it
    # - Define its losses and maybe add it to loss_ae
    # - Add the loss to tf summary
    # - Optimize it and add that to `train_ops`
    # - Add model to `save_models` so it will end up being saved.
    # Except for pretained models which do not need to be trained or saved

    print(hvd.rank(), "building model and losses...", flush=True)
    with tf.name_scope("autoencoder"):
        with tf.device("/cpu:0"):
            encoder = load_model_def(FLAGS.model_dir, "encoder")
            decoder = load_model_def(FLAGS.model_dir, "decoder")
            if not encoder or not decoder:
                print("Defining New encoder and decoder.")
                encoder, decoder = models.autoencoder(
                    shape=shape,
                    base=FLAGS.base_dim,
                    batchnorm=FLAGS.batchnorm,
                    n_blocks=FLAGS.n_blocks,
                    variational=FLAGS.variational,
                    dense=FLAGS.dense_ae,
                    block_len=FLAGS.block_len,
                    scale=FLAGS.scale,
                    data_format=FLAGS.channel_order,
                )

    with tf.name_scope("noise"):
        noised_img = add_noise(img, FLAGS.salt_pepper, FLAGS.gaussian_noise)
        if noised_img is not img:
            tf.summary.image("noised_image", cmap(noised_img), FLAGS.display_imgs)

    if FLAGS.variational:
        z, latent_mean, latent_logvar = encoder(noised_img)
        with tf.name_scope("kl_div"):
            kl_div = -0.5 * tf.reduce_mean(
                1 + latent_logvar - latent_mean ** 2 - tf.exp(latent_logvar)
            )
            tf.summary.scalar("kl_div", kl_div)
        loss_ae = FLAGS.vae_beta * kl_div

    else:
        z = encoder(noised_img)
        with tf.name_scope("bottleneck"):
            tf.summary.histogram("histogram", z)
        loss_ae = 0

    ae_img = decoder(z)
    loss_ae += image_losses(img, ae_img, *FLAGS.image_loss_weights)

    cimg = cmap(img)
    camg = cmap(ae_img)
    tf.summary.image("original", cimg, FLAGS.display_imgs)
    tf.summary.image("autoencoded", camg, FLAGS.display_imgs)
    tf.summary.image("difference", cimg - camg, FLAGS.display_imgs)
    save_models = {"encoder": encoder, "decoder": decoder}

    train_ops = []

    if FLAGS.adversarial:
        with tf.name_scope("discriminator"):
            with tf.device("/cpu:0"):
                disc = load_model(FLAGS.model_dir, "disc")
                if not disc:
                    disc = models.discriminator(shape, FLAGS.n_blocks)

        z_noise = tf.random_normal(z.shape)
        gen_img = decoder(z_noise)

        with tf.name_scope("disc_loss"):
            di = tf.reduce_mean(disc(img))
            da = tf.reduce_mean(disc(ae_img))
            dg = tf.reduce_mean(disc(gen_img))
            loss_disc = 2 * di - da - dg
            tf.summary.scalar("loss_disc", loss_disc)
            tf.summary.scalar("disc_ae_img", da)
            tf.summary.scalar("disc_gen", dg)

        tf.summary.image("generated", cmap(gen_img), FLAGS.display_imgs)
        save_models["disc"] = disc
        loss_ae += FLAGS.lambda_disc * tf.reduce_mean(da + dg)
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
        dsc_optimizer = new_adam_optimizer(hvd.size(), *FLAGS.discriminator_adam)
        train_disc = dsc_optimizer.minimize(loss_disc, var_list=disc.trainable_weights)
        train_ops.append(train_disc)

    if FLAGS.perceptual:
        # There is a minimum shape thats 139 or so but we only need early layers
        # Set input height / width to None so Keras doesn't complain
        inp_shape = None, None, 3
        per = our_models.classifier(inp_shape)
        with tf.name_scope("perceptual_loss"):
            loss_per = tf.reduce_mean(tf.square(per(cimg) - per(camg)))
            loss_ae += loss_per * FLAGS.lambda_per
            tf.summary.scalar("loss_per", loss_per)

    # Monitor AE gradients
    ae_optimizer = new_adam_optimizer(hvd.size(), *FLAGS.autoencoder_adam)

    with tf.name_scope("grad_info"):
        grads_and_vars = ae_optimizer.compute_gradients(
            loss_ae, var_list=encoder.trainable_weights + decoder.trainable_weights
        )
        for grad, var in grads_and_vars:
            if grad is not None and not FLAGS.no_grad_histogram:
                tf.summary.histogram("{}/histogram".format(var.name), grad)
    train_ops.append(ae_optimizer.apply_gradients(grads_and_vars, global_step))
    tf.summary.scalar("loss_ae", loss_ae)

    # Save model definitions
    for m in save_models:
        with open(path.join(FLAGS.model_dir, f"{m}.json"), "w") as f:
            f.write(save_models[m].to_json())

    summary_op = tf.summary.merge_all()
    run_metadata = tf.RunMetadata()
    run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    print("Training...", flush=True)
    with tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                allow_growth=True, visible_device_list=str(hvd.local_rank())
            ),
            log_device_placement=FLAGS.log_device_placement,
        )
    ) as sess:
        profiler = Profiler(sess.graph)
        sess.run(
            tf.global_variables_initializer(),
            options=run_opts,
            run_metadata=run_metadata,
        )
        hvd.broadcast_global_variables(0)

        for m in save_models:
            try:
                gs = load_latest_model_weights(save_models[m], FLAGS.model_dir, m)
                sess.run(global_step.assign(gs))
                print("Loaded weights for", m, "Set global step to", gs, flush=True)
            except Exception as e:
                print("Could not load weights for", m, "because", e, flush=True)

        log_dir = path.join(FLAGS.model_dir, "summary")
        tb_graph = sess.graph if not path.exists(log_dir) else None
        summary_writer = tf.summary.FileWriter(log_dir, graph=tb_graph)

        if hvd.rank() == 0:
            for m in save_models:
                save_models[m].summary()
            print("", flush=True)

        for _ in range(FLAGS.max_steps):
            gs, _ = sess.run(
                [global_step, train_ops], options=run_opts, run_metadata=run_metadata
            )

            if gs % FLAGS.summary_every == 0 and hvd.rank() == 0:
                summary = sess.run(
                    summary_op, options=run_opts, run_metadata=run_metadata
                )
                summary_writer.add_run_metadata(run_metadata, "step%d" % gs)
                summary_writer.add_summary(summary, gs)
                summary_writer.flush()
                profiler.add_step(int(gs), run_metadata)
                timeline_json = path.join(FLAGS.model_dir, "timelines", "t.json")
                profiler.profile_graph(
                    options=(
                        ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory())
                        .with_step(gs)
                        .with_timeline_output(timeline_json)
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

            if gs % FLAGS.save_every == 0 and hvd.rank() == 0:
                for m in save_models:
                    save_models[m].save_weights(
                        path.join(FLAGS.model_dir, f"{m}-{gs}.h5")
                    )
