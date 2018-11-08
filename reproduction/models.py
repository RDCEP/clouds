"""models.py: Define all the autoencoder variancts used in the clouds project.
The main function is `autoencoder` however there is also a discriminator and a perceptual
network for various losses.
"""
__author__ = "casperneo@uchicago.edu"

import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as pretrained

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential


def sample_variational(x, depth, dense, nonlinearity, data_format):
    if dense:
        sh = x.shape.as_list()[1:]
        x = Flatten()(x)
        mn = Dense(dense, name="latent_mean")(x)
        lv = Dense(dense, name="latent_log_var", kernel_initializer="zeros")(x)
    else:
        mn = Conv2D(depth, 1, data_format=data_format)(x)
        mn = nonlinearity()(mn)
        mn = Conv2D(depth, 1, data_format=data_format)(mn)
        lv = Conv2D(depth, 1, data_format=data_format)(x)
        lv = nonlinearity()(lv)
        lv = Conv2D(depth, 1, data_format=data_format)(lv)

    x = Lambda(
        lambda arg: tf.random_normal(arg[0].shape[1:]) * tf.exp(arg[1] / 2) + arg[0],
        name="sampling",
    )([mn, lv])

    if dense:
        x = Dense(np.product(sh))(x)
        x = Reshape(sh)(x)

    return mn, lv, x


def residual_add(x, r, data_format):
    """Adds `r` to `x` reshaping `r` and zero_padding extra dimensions
    If the shape doesn't fit, the input tensor is bilinear resized
    If the input tensor has too many channels only the first channels will be added
    If the input tensor has too few channels it will be added to the first output chnanels
    """

    def fn(args):
        # NOTE need to reimport to successfully reload models
        import tensorflow as tf
        from functools import partial

        x, r = args
        x_shape, r_shape = tf.shape(x), tf.shape(r)

        if data_format == "channels_last":
            r_c, x_c = r_shape[3], x_shape[3]
            r_hw, x_hw = r_shape[1:3], x_shape[1:3]

            zero_pad_channel = lambda r: tf.pad(
                r, [[0, 0], [0, 0], [0, 0], [0, x_c - r_c]]
            )
            slice_channel = lambda r: r[:, :, :, :x_c]
            resize = lambda r: tf.image.resize_bilinear(r, x_hw)

        elif data_format == "channels_first":
            r_c, x_c = r_shape[1], x_shape[1]
            r_hw, x_hw = r_shape[2:4], x_shape[2:4]

            def zero_pad_channel(r):
                return tf.pad(r, [[0, 0], [0, x_c - r_c], [0, 0], [0, 0]])

            slice_channel = lambda r: r[:, :x_c]

            def resize(r):
                # HACK this is certainly not efficient
                r = tf.transpose(r, [0, 3, 1, 2])
                r = tf.image.resize_bilinear(r, x_hw)
                return tf.transpose(r, [0, 3, 1, 2])

        else:
            raise ValueError("Invalid data_format: `%s`" % data_format)

        r = tf.cond(tf.greater(r_c, x_c), partial(slice_channel, r), lambda: r)
        r = tf.cond(tf.less(r_c, x_c), partial(zero_pad_channel, r), lambda: r)
        r = tf.cond(
            tf.reduce_any(tf.not_equal(r_hw, x_hw)), partial(resize, r), lambda: r
        )

        return x + r

    return Lambda(fn)([x, r])


def resblock(x, depth, nonlinearity, data_format):
    """Two 3x3 convoluions with a residual addition between them
    """
    r = x
    x = nonlinearity()(x)
    x = Conv2D(depth, 3, padding="same", data_format=data_format)(x)
    x = nonlinearity()(x)
    x = Conv2D(depth, 3, padding="same", data_format=data_format)(x)
    return residual_add(x, r, data_format)


def resblocks(x, depth, blocks, nonlinearity, data_format, cardinality=1):
    """Repeatedly applies resblock, supports ResNeXT caridnality
    would be faster if there were grouped convolution ops in tf.
    """
    if cardinality == 1:
        for _ in range(blocks):
            x = resblock(x, depth, nonlinearity, data_format)
    else:
        lanes = []
        for _ in range(cardinality):
            b = x
            for _ in range(blocks):
                b = resblock(b, depth, nonlinearity, data_format)
            lanes.append(b)
        x = Add()(lanes)

    return x


def scale_change_block(
    x, depth, nonlinearity, down, scale=2, data_format="channels_first"
):
    """2 3x3 convs with nonlinearities, first one is strided (and ConvT if not down).
    """
    r = x
    x = nonlinearity()(x)
    _conv = Conv2D if down else Conv2DTranspose
    x = _conv(depth, max(3, scale), scale, padding="same", data_format=data_format)(x)
    x = nonlinearity()(x)
    x = Conv2D(depth, 3, padding="same", data_format=data_format)(x)
    return residual_add(x, r, data_format)


def autoencoder(
    shape,
    depths,
    batchnorm=True,
    variational=False,
    dense=None,
    block_len=0,
    nonlinearity=LeakyReLU,
    scale=2,
    data_format="channels_first",
):
    """Defines a convolutional autoencoder, variational if specified

    The autoencoder has the following properties:
        Same number of blocks in encoder and decoder
        Small, 3x3 kernels
        Stride `scale=2` convolutions and transposed convolutions
        Skip connections from inputs to outputs of every block
    Args:
        shape: (height, width, channels) of input image (chann)
        depths: Number of channels of the blocks in the encoder. `reversed(depths)` will
            represent the number of channels of the blocks in the decoder.
        base: Number of channels of in first block, each subsequent block doubles channels
            until the bottelneck layer, then each block halves in channels until the image
            is decoded
        batchnorm: Use batchnorm layer at the end of every block
        variational: Use a variational autoencoder - bottleneck layer output is a mean and
            standard deviation. A sample from that distribution is decoded
        dense: Integer specifying dimension of bottleneck layer
        block_len: Number of extra convolutions performed in each block, in addition to
            the scale change block
        nonlinearity: Function that returns a keras activation layer
        scale: Factor to change scales with in each scale change block
        data_format: "channels_last" or "channels_first"
    Returns:
        Two keras models: an encoder model and a decoder model.
    """
    # Encoder
    x = inp = Input(shape=shape, name="encoder_input")

    x = Conv2D(depths[0], 3, padding="same", data_format=data_format)(x)
    x = resblocks(x, depths[0], block_len, nonlinearity, data_format)
    for i, depth in enumerate(depths):
        with tf.variable_scope("encoding_%d" % i):
            # Half image size
            x = scale_change_block(
                x, depth, nonlinearity, down=True, scale=scale, data_format=data_format
            )
            x = resblocks(x, depth, block_len, nonlinearity, data_format=data_format)
            if batchnorm:
                x = BatchNormalization()(x)

    if variational:
        with tf.variable_scope("sample_variational"):
            mn, lv, x = sample_variational(x, depth, dense, nonlinearity, data_format)
        encoder = Model(inp, [x, mn, lv], name="encoder")
    else:
        if dense:
            sh = [int(s) for s in x.shape[1:]]
            x = Flatten()(x)
            x = Dense(dense)(x)
            x = nonlinearity()(x)

        encoder = Model(inp, x)

    # Decoder
    x = inp = Input(x.shape[1:], name="decoder_input")
    if dense:
        x = Dense(np.prod(sh))(x)
        x = nonlinearity()(x)
        x = Reshape(sh)(x)

    for i, depth in enumerate(reversed(depths)):
        with tf.variable_scope("decoding_%d" % i):
            # Double Image size
            x = scale_change_block(
                x, depth, nonlinearity, down=False, scale=scale, data_format=data_format
            )
            x = resblocks(x, depth, block_len, nonlinearity, data_format=data_format)
            if batchnorm:
                x = BatchNormalization()(x)

    input_channels = shape[0] if data_format == "channels_first" else shape[2]
    x = nonlinearity()(x)
    x = Conv2D(input_channels, 1, name="reconstructed", data_format=data_format)(x)
    decoder = Model(inp, x, name="decoder")

    return encoder, decoder


def discriminator(
    shape, n_layers=3, base=8, nonlinearity=LeakyReLU, data_format="channels_first"
):
    """Image -> probability network
    """
    x = inp = Input(shape=shape, name="disc_input")

    for i in range(n_layers):
        depth = base * 2 ** i
        x = Conv2D(depth, 3, 2, padding="same", data_format=data_format)(x)
        x = nonlinearity()(x)
        x = Conv2D(depth, 3, padding="same", data_format=data_format)(x)
        x = nonlinearity()(x)
        x = BatchNormalization()(x)

    x = Conv2D(1, 1, data_format=data_format)(x)
    x = GlobalAveragePooling2D()(x)

    return Model(inp, x)


def classifier(shape, layer_name="mixed5"):
    # 1 4 3 RGB
    p = pretrained.InceptionV3(include_top=False, input_shape=shape)
    out = p.get_layer(layer_name).output
    return Model(p.inputs, out)
