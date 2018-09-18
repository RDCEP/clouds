import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
import tensorflow.keras.applications as pretrained
import numpy as np


def sample_variational(x, depth, dense):
    if dense:
        sh = x.shape.as_list()[1:]
        x = Flatten()(x)
        mn = Dense(depth, name="latent_mean")(x)
        lv = Dense(depth, name="latent_log_var", kernel_initializer="zeros")(x)
    else:
        mn = Conv2D(depth, 1, activation="relu")(x)
        mn = Conv2D(depth, 1)(mn)
        lv = Conv2D(depth, 1, activation="relu")(x)
        lv = Conv2D(depth, 1)(lv)

    x = Lambda(
        lambda arg: tf.random_normal(arg[0].shape[1:]) * tf.exp(arg[1] / 2) + arg[0],
        name="sampling",
    )([mn, lv])

    if dense:
        x = Dense(np.product(sh))(x)
        x = Reshape(sh)(x)

    return mn, lv, x


def residual_add(x, r):
    """Adds `r` to `x` reshaping `r` and zero_padding extra dimensions.
    """

    def fn(args):
        x, r = args
        _, h, w, c = x.shape

        if r.shape[1:3] != [h, w]:
            r = tf.image.resize_bilinear(r, x.shape[1:3])

        if r.shape[3] >= c:
            return x + r[:, :, :, :c]
        else:
            return x + tf.pad(r, [[0, 0], [0, 0], [0, 0], [0, c - r.shape[3]]])

    return Lambda(fn)([x, r])


def resblock3(x, depth, length=3):
    r = x
    x = Conv2D(depth // 4, 1, activation="relu")(x)
    x = Conv2D(depth, 3, padding="same", activation="relu")(x)
    x = Conv2D(depth, 1)(x)
    return residual_add(x, r)


def resblock2(x, depth):
    r = x
    x = Conv2D(depth, 3, padding="same", activation="relu")(x)
    x = Conv2D(depth, 3, padding="same", activation="relu")(x)
    return residual_add(x, r)


def resblocks(x, depth, blocks, cardinality=1):
    """Repeats several resblock2 or resblock3, the latter used when depth is high to
    conserve memory. Either way there are 2 non-linearities. TODO renext cardinality.
    """
    if cardinality == 1:
        for _ in range(blocks):
            b = resblock3(b, depth) if depth >= 256 else resblock2(x, depth)
    else:
        lanes = []
        for _ in range(cardinality):
            b = x
            for _ in range(blocks):
                b = resblock3(b, depth) if depth >= 256 else resblock2(x, depth)
            lanes.append(b)
        x = Add()(lanes)

    return x


def scale_change_block(x, depth, down, length=2, name=None):
    r = x
    _conv = Conv2D if down else Conv2DTranspose
    x = _conv(depth, 3, 2, activation="relu", padding="same")(x)
    x = Conv2D(depth, 3, activation="relu", padding="same")(x)
    return residual_add(x, r)


def autoencoder(
    shape, n_blocks, base, batchnorm, variational, dense=False, block_len=1
):
    """
    Returns an encoder model and autoencoder model
    """
    x = inp = Input(shape=shape, name="ae_input")
    outputs = []

    # Encoder
    x = Conv2D(base, 3, activation="relu", padding="same")(x)
    x = resblocks(x, base, block_len)
    for i in range(n_blocks):
        with tf.variable_scope("encoding_%d" % i):
            depth = base * 2 ** i
            # Half image size
            x = scale_change_block(x, depth, down=True)
            x = resblocks(x, depth, block_len)
            if batchnorm:
                x = BatchNormalization()(x)

    # Hidden vector
    if variational:
        with tf.variable_scope("sample_variational"):
            mn, lv, x = sample_variational(x, depth, dense)
            outputs.extend([mn, lv])
    outputs.append(x)

    # Decoder
    for i in range(n_blocks - 1, -1, -1):
        with tf.variable_scope("decoding_%d" % i):
            depth = base * 2 ** i
            # Double Image size
            x = scale_change_block(x, depth, down=False)
            x = resblocks(x, depth, block_len)
            if batchnorm:
                x = BatchNormalization()(x)

    x = Conv2D(shape[-1], 1, name="reconstructed")(x)
    outputs.append(x)

    return Model(inp, outputs)


def discriminator(shape, n_layers=3):
    """Image -> probability network
    """
    x = inp = Input(shape=shape, name="disc_input")

    for i in range(n_layers):
        depth = 32 * 2 ** i
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
        x = Conv2D(depth, 3, 2, activation="relu", padding="same")(x)

    x = Conv2D(1, 1)(x)
    x = GlobalAveragePooling2D()(x)

    return Model(inp, x)


def classifier(shape, layer_name="mixed5"):
    # 1 4 3 RGB
    p = pretrained.InceptionV3(include_top=False, input_shape=shape)
    out = p.get_layer(layer_name).output
    return Model(p.inputs, out)
