import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
import tensorflow.keras.applications as pretrained


def autoencoder(shape, n_layers=3, base=32, variational=False):
    """
    Returns an encoder model and autoencoder model
    """
    x = inp = Input(shape=shape, name="ae_input")
    outputs = []

    # Encoder
    for i in range(n_layers):
        depth = base * 2 ** i
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
        x = Conv2D(depth, 3, 2, activation="relu", padding="same")(x)

    if variational:
        mn = Conv2D(depth, 1, name="latent_mean")(x)
        lv = Conv2D(depth, 1, name="latent_log_var", kernel_initializer="zeros")(x)
        x = Lambda(
            lambda arg: tf.random_normal(arg[0].shape[1:]) * tf.exp(arg[1] / 2)
            + arg[0],
            name="sampling",
        )([mn, lv])
        outputs.extend([mn, sd])

    # Hidden vector
    outputs.append(x)

    # Decoder
    for i in range(n_layers):
        depth = base * 2 ** (n_layers - i - 1)
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
        x = Conv2DTranspose(depth, 3, 2, activation="relu", padding="same")(x)

    x = Conv2D(shape[-1], 3, activation="relu", padding="same", name="reconstructed")(x)
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


def dilated_ae(shape, n_layers=3):
    x = inp = Input(shape=shape, name="ae_input")

    # Encoder
    for i in range(n_layers):
        depth = 32 * 2 ** i
        x = Conv2D(depth, 3, dilation_rate=2 ** i, activation="relu", padding="same")(x)

    encoded = x

    # Decoder
    for i in range(n_layers, -1, -1):
        depth = 32 * 2 ** i
        x = Conv2DTranspose(
            depth, 3, dilation_rate=2 ** i, activation="relu", padding="same"
        )(x)

    decoded = Conv2D(shape[-1], 3, activation="relu", padding="same")(x)

    return Model(inp, encoded), Model(inp, decoded)
