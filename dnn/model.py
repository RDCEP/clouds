import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential


def autoencoder(shape, n_layers=3):
    """
    Returns an encoder model and autoencoder model
    """
    x = inp = Input(shape=shape, name="ae_input")

    # Encoder
    for i in range(n_layers):
        depth = 32 * 2 ** i
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
        x = Conv2D(depth, 3, 2, activation="relu", padding="same")(x)
        # x = MaxPool2D(2, padding="same")(x)

    encoded = x

    # Decoder
    for i in range(n_layers):
        depth = 32 * 2 ** (n_layers - i - 1)
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
        x = Conv2DTranspose(depth, 3, 2, activation="relu", padding="same")(x)
        # x = UpSampling2D(2)(x)

    decoded = Conv2D(shape[-1], 3, activation="relu", padding="same")(x)

    return Model(inp, encoded), Model(inp, decoded)


def discriminator(shape, n_layers=3):
    """Image -> probability network
    """
    x = inp = Input(shape=shape, name="disc_input")

    for i in range(n_layers):
        depth = 32 * 2 ** i
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
        x = Conv2D(depth, 3, 2, activation="relu", padding="same")(x)

    x = Conv2D(1, 3)(x)
    x = GlobalAveragePooling2D(x)

    return Model(inp, x)


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
