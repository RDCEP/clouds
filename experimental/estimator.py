import tensorflow as tf
from models import autoencoder
from train.lib import *


def model_fn(features, labels, mode, params):
    del labels  # unused
    img = features
    FLAGS = params["flags"]

    cmap = ColorMap(FLAGS.red_bands, FLAGS.green_bands, FLAGS.blue_bands)

    with tf.name_scope("autoencoder"):
        encoder = load_model(FLAGS.model_dir, "encoder")
        decoder = load_model(FLAGS.model_dir, "decoder")
        if not encoder or not decoder:
            encoder, decoder = models.autoencoder(
                shape=FLAGS.shape,
                base=FLAGS.base_dim,
                batchnorm=FLAGS.batchnorm,
                n_blocks=FLAGS.n_blocks,
                variational=FLAGS.variational,
                block_len=FLAGS.block_len,
                scale=FLAGS.scale,
                data_format=FLAGS.channel_order,
            )

    encoder.summary()
    decoder.summary()

    with tf.name_scope("noise"):
        noised_img = add_noise(img, FLAGS.salt_pepper, FLAGS.gaussian_noise)
        if noised_img is not img:
            tf.summary.image("noised_image", cmap(noised_img), FLAGS.display_imgs)

    if FLAGS.variational:
        z, latent_mean, latent_logvar = encoder(noised_img)
        ae_img = decoder(z)
        kl_div = lambda: -0.5 * tf.reduce_mean(
            1 + latent_logvar - latent_mean ** 2 - tf.exp(latent_logvar)
        )
        loss_ae += loss_fn("kl_div", FLAGS.vae_beta, kl_div)

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
        optimizers = {}
        train_ops = []

        if FLAGS.adversarial:
            with tf.name_scope("discriminator"):
                with tf.device("/cpu:0"):
                    disc = load_model(FLAGS.model_dir, "disc")
                    if not disc:
                        disc = models.discriminator(shape, FLAGS.n_blocks)
                if FLAGS.num_gpu > 1:
                    disc = tf.keras.utils.multi_gpu_model(disc, FLAGS.num_gpu)

            z_noise = tf.random_normal(tf.shape(z))
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
            dsc_optimizer = tf.train.AdamOptimizer(*FLAGS.discriminator_adam)
            train_disc = dsc_optimizer.minimize(
                loss_disc, var_list=disc.trainable_weights
            )
            optimizers["dsc_optimizer"] = dsc_optimizer
            train_ops.append(train_disc)

    if FLAGS.perceptual:
        # There is a minimum shape thats 139 or so but we only need early layers
        # Set input height / width to None so Keras doesn't complain
        inp_shape = None, None, 3
        per = our_models.classifier(inp_shape)
        if FLAGS.num_gpu > 1:
            per = tf.keras.utils.multi_gpu_model(per, FLAGS.num_gpu)
        with tf.name_scope("perceptual_loss"):
            pi = per(cmap(img))
            pa = per(cmap(ae_img))
            loss_per = tf.reduce_mean(tf.square(pi - pa))
            loss_ae += loss_per * FLAGS.lambda_per
            tf.summary.scalar("loss_per", loss_per)

    # Monitor AE gradients
    ae_optimizer = tf.train.AdamOptimizer(*FLAGS.autoencoder_adam)
    with tf.name_scope("grad_info"):
        grads_and_vars = ae_optimizer.compute_gradients(
            loss_ae, var_list=encoder.trainable_weights + decoder.trainable_weights
        )
        for grad, var in grads_and_vars:
            if grad is not None:
                if not FLAGS.no_grad_histogram:
                    tf.summary.histogram("{}/histogram".format(var.name), grad)
    train_ops.append(ae_optimizer.apply_gradients(grads_and_vars))
    optimizers["ae_optimizer"] = ae_optimizer
    tf.summary.scalar("loss_ae", loss_ae)

    train_ops.append(tf.assign_add(tf.train.get_or_create_global_step(), 1))
    train_ops = tf.group(*train_ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_ae, train_op=train_ops)

    if mode == tf.estimator.ModeKeys.EVAL:
        raise NotImplementedError

    raise ValueError("Invalid mode", mode)


def input_fn(FLAGS):
    def fn():
        with tf.name_scope("dataset"):
            dataset = pipeline.load_data(
                FLAGS.data,
                FLAGS.shape,
                FLAGS.batch_size,
                FLAGS.read_threads,
                FLAGS.shuffle_buffer_size,
                FLAGS.prefetch,
                not FLAGS.no_augment_flip,
                not FLAGS.no_augment_rotate,
            )
        return dataset.map(lambda names, coords, imgs: (imgs, (names, coords)))

    return fn


FLAGS = get_flags()

tf.estimator.Estimator(model_fn, params={"flags": FLAGS}).train(
    input_fn(FLAGS),
    hooks=[
        tf.train.StepCounterHook(),
        tf.train.CheckpointSaverHook(
            path.join(FLAGS.model_dir, "ckpts"), save_secs=600
        ),
        tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(), save_secs=300),
        # tf.train.SummarySaverHook(save_secs=300),
        tf.train.ProfilerHook(save_secs=600),
    ],
)
