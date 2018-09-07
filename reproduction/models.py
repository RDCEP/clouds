import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
import tensorflow.keras.applications as pretrained
import numpy as np

def resblock(x, depth, block_len = 2):
    if not block_len: return x
    r = x
    for _ in range(block_len):
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
    return Add()([r,x])


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
        lambda arg: tf.random_normal(arg[0].shape[1:]) * tf.exp(arg[1] / 2)
        + arg[0],
        name="sampling",
    )([mn, lv])

    if dense:
        x = Dense(np.product(sh))(x)
        x = Reshape(sh)(x)

    return mn, lv, x


def autoencoder(shape, n_blocks, base, batchnorm, variational, dense=False, block_len=2):
    """
    Returns an encoder model and autoencoder model
    """
    x = inp = Input(shape=shape, name="ae_input")
    outputs = []

    # Encoder
    for i in range(n_blocks):
        depth = base * 2 ** i
        x = resblock(x, depth, block_len)
        # Half image size
        x = Conv2D(depth, 3, 2, activation="relu", padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)

    # Hidden vector
    if variational:
        mn, lv, x = sample_variational(x, depth, dense)
        outputs.extend([mn, lv])
    outputs.append(x)

    # Decoder
    for i in range(n_blocks -1, -1, -1):
        depth = base * 2 ** i
        x = resblock(x, depth, block_len)
        # Double Image size
        x = Conv2DTranspose(depth, 3, 2, activation="relu", padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)

    x = Conv2D(shape[-1], 1, activation="relu", padding="same", name="reconstructed")(x)
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
